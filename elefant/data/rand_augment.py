import math, random, torch
import time
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict, Any, Optional
from torch import nn
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)
from kornia.color import hls_to_rgb, rgb_to_hls
from kornia.augmentation import random_generator as rg
from kornia.geometry.transform import get_perspective_transform, warp_perspective
from math import pi
from kornia.augmentation import AugmentationBase2D
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from torch import Tensor


class FastRandomErasing(AugmentationBase2D):
    """
    Fast single-rectangle erasing (cutout) using one in-place slice across the batch
    """

    def __init__(
        self,
        scale=(0.02, 0.1),
        ratio=(0.3, 3.3),
        value: float = 0.0,
        p: float = 0.35,
        same_on_batch: bool = True,
        keepdim: bool = False,
    ):
        super().__init__(p=p, p_batch=1.0, same_on_batch=same_on_batch, keepdim=keepdim)
        self.scale = scale
        self.ratio = ratio
        self.value = float(value)
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    @staticmethod
    def _sample_rect(H: int, W: int, scale, log_ratio, device):
        area = H * W
        for _ in range(10):
            target_area = (
                torch.empty((), device=device).uniform_(scale[0], scale[1]) * area
            )
            aspect = torch.exp(
                torch.empty((), device=device).uniform_(log_ratio[0], log_ratio[1])
            )
            h = torch.clamp((target_area / aspect).sqrt().round().to(torch.long), 1, H)
            w = torch.clamp((target_area * aspect).sqrt().round().to(torch.long), 1, W)
            if (h <= H) and (w <= W):
                top = torch.randint(0, H - int(h) + 1, (1,), device=device).item()
                left = torch.randint(0, W - int(w) + 1, (1,), device=device).item()
                return top, left, int(h), int(w)
        # Fallback: small centered box
        h = max(1, int(H * scale[0] ** 0.5))
        w = max(1, int(W * scale[0] ** 0.5))
        top = (H - h) // 2
        left = (W - w) // 2
        return top, left, h, w

    def apply_transform(self, x: Tensor, params, flags, transform=None) -> Tensor:
        N, C, H, W = x.shape
        top, left, h, w = self._sample_rect(H, W, self.scale, self.log_ratio, x.device)
        # In-place fill for the entire batch
        x[:, :, top : top + h, left : left + w] = self.value
        return x


class FastRandomResizedCrop(AugmentationBase2D):
    """
    Fast RandomResizedCrop using a single grid_sample for the whole batch.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        scale=(0.5, 1.0),
        ratio=(0.5, 2.0),
        interpolation: str = "bilinear",
        p: float = 0.35,
        same_on_batch: bool = True,
        keepdim: bool = False,
    ):
        super().__init__(p=p, p_batch=1.0, same_on_batch=same_on_batch, keepdim=keepdim)
        self.out_h, self.out_w = size
        self.scale = scale
        self.ratio = ratio
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        self.interpolation = interpolation

    @staticmethod
    def _pick_crop(H: int, W: int, scale, log_ratio, device):
        area = H * W
        for _ in range(10):
            target_area = (
                torch.empty((), device=device).uniform_(scale[0], scale[1]) * area
            )
            aspect = torch.exp(
                torch.empty((), device=device).uniform_(log_ratio[0], log_ratio[1])
            )
            crop_h = torch.clamp(
                (target_area / aspect).sqrt().round().to(torch.long), 1, H
            )
            crop_w = torch.clamp(
                (target_area * aspect).sqrt().round().to(torch.long), 1, W
            )
            if (crop_h <= H) and (crop_w <= W):
                top = torch.randint(0, H - int(crop_h) + 1, (1,), device=device).item()
                left = torch.randint(0, W - int(crop_w) + 1, (1,), device=device).item()
                return top, left, int(crop_h), int(crop_w)
        # Fallback center crop honoring ratio bounds
        in_ratio = W / H
        if in_ratio < math.exp(log_ratio[0]):
            crop_w = W
            crop_h = int(round(crop_w / math.exp(log_ratio[0])))
        elif in_ratio > math.exp(log_ratio[1]):
            crop_h = H
            crop_w = int(round(crop_h * math.exp(log_ratio[1])))
        else:
            crop_h, crop_w = H, W
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        return top, left, int(crop_h), int(crop_w)

    def apply_transform(self, x: Tensor, params, flags, transform=None) -> Tensor:
        N, C, H, W = x.shape
        device = x.device

        top, left, crop_h, crop_w = self._pick_crop(
            H, W, self.scale, self.log_ratio, device
        )

        # Build one sampling grid and expand across batch (align_corners=False)
        xs = torch.linspace(
            left + 0.5, left + crop_w - 0.5, self.out_w, device=device, dtype=x.dtype
        )
        ys = torch.linspace(
            top + 0.5, top + crop_h - 0.5, self.out_h, device=device, dtype=x.dtype
        )
        gx = xs.view(1, self.out_w).expand(self.out_h, self.out_w)
        gy = ys.view(self.out_h, 1).expand(self.out_h, self.out_w)
        gx = gx / W * 2 - 1
        gy = gy / H * 2 - 1
        grid = (
            torch.stack((gx, gy), dim=-1)
            .unsqueeze(0)
            .expand(N, -1, -1, -1)
            .contiguous()
        )

        mode = "bilinear" if self.interpolation == "bilinear" else "nearest"
        out = F.grid_sample(
            x, grid, mode=mode, padding_mode="zeros", align_corners=False
        )
        return out


class ColorJiggle(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a tensor image.

    This version has been rewritten so that the order of operations is extracted as static Python
    integers—avoiding data-dependent indexing that TorchDynamo struggles with.
    """

    def __init__(
        self,
        brightness: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        same_on_batch: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        assert same_on_batch, (
            "same_on_batch should always be True to keep augmentations consistent with different video frames"
        )
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._param_generator = rg.ColorJiggleGenerator(
            brightness, contrast, saturation, hue
        )

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        order_tensor = params["order"]
        order = [order_tensor[i].item() for i in range(order_tensor.numel())]

        brightness_factor = params["brightness_factor"]
        contrast_factor = params["contrast_factor"]
        saturation_factor = params["saturation_factor"]
        hue_factor = params["hue_factor"]

        def brightness_op(img: Tensor) -> Tensor:
            if (brightness_factor - 1).any():
                return adjust_brightness(img, brightness_factor - 1)
            return img

        def contrast_op(img: Tensor) -> Tensor:
            if (contrast_factor != 1).any():
                return adjust_contrast(img, contrast_factor)
            return img

        def saturation_op(img: Tensor) -> Tensor:
            if (saturation_factor != 1).any():
                return adjust_saturation(img, saturation_factor)
            return img

        def hue_op(img: Tensor) -> Tensor:
            if (hue_factor != 0).any():
                return adjust_hue(img, hue_factor * 2 * pi)
            return img

        ops = [brightness_op, contrast_op, saturation_op, hue_op]

        jittered = input
        for idx in order:
            jittered = ops[idx](jittered)
        return jittered


class isoNoise(AugmentationBase2D):
    def __init__(self, p=0.5, p_batch=1.0, same_on_batch=True, keepdim=False):
        super().__init__(p, p_batch, same_on_batch, keepdim)
        assert same_on_batch, (
            "same_on_batch should always be True to keep augmentations consistent with different video frames"
        )
        # Pre-generate parameter ranges
        self.color_shift_range = (0.01, 0.2)
        self.intensity_range = (0.1, 0.6)

    def apply_transform(self, x, params, flags, transform=None):
        # Generate parameters once per batch
        color_shift = (
            torch.empty(1, device=x.device).uniform_(*self.color_shift_range).item()
        )
        intensity = (
            torch.empty(1, device=x.device).uniform_(*self.intensity_range).item()
        )

        # Convert to HLS once
        hls_img = rgb_to_hls(x)

        # Compute luminance noise more efficiently
        l_channel = hls_img[:, 1]
        l_std = l_channel.flatten(1).std(-1, keepdim=True).unsqueeze(-1)
        luminance_scale = (l_std * intensity * 255).expand(-1, x.size(-2), x.size(-1))
        luminance_noise = torch.poisson(luminance_scale) / 255

        # Apply color noise
        color_noise = torch.randn(1, x.size(-2), x.size(-1), device=x.device) * (
            color_shift * 2 * pi * intensity
        )

        # Update channels in-place
        hls_img[:, 0] = (hls_img[:, 0] + color_noise) % (2 * pi)
        hls_img[:, 1] = hls_img[:, 1] + luminance_noise * (1.0 - hls_img[:, 1])

        return hls_to_rgb(hls_img)


class Blur(AugmentationBase2D):
    def __init__(self, blur_prob, gaussian_blur_weight):
        super().__init__()
        self.blur_prob = blur_prob
        self.gaussian_blur_weight = gaussian_blur_weight
        self.gaussian_blur = K.RandomGaussianBlur(
            kernel_size=(3, 7),
            sigma=(0.1, 2.0),
            p=1.0,
            same_on_batch=True,
        )
        self.motion_blur = K.RandomMotionBlur(
            kernel_size=(3, 7),
            angle=(0.0, 360.0),
            direction=(-1.0, 1.0),
            p=1.0,
            same_on_batch=False,
        )

    def apply_transform(self, x, params, flags, transform=None):
        if torch.rand(1) < self.blur_prob:
            if torch.rand(1) < self.gaussian_blur_weight:
                x = self.gaussian_blur(x)
            else:
                x = self.motion_blur(x)
        return x


class SpatialTransformBatch(AugmentationBase2D):
    def __init__(self):
        super().__init__(p=1.0, p_batch=1.0, same_on_batch=True, keepdim=False)

    def spatial_transform_batch(self, x):
        B, C, H, W = x.shape
        device = x.device

        # Pre-allocate tensors
        coordinates_original = (
            torch.tensor(
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], device=device
            )
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        # Generate random angles more efficiently
        angles_rad = torch.empty(1, device=device).uniform_(-3 * pi / 180, 3 * pi / 180)

        # Vectorized rotation matrix creation
        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)

        rot_mats = torch.zeros(B, 2, 2, device=device)
        rot_mats[:, 0, 0] = cos_a
        rot_mats[:, 0, 1] = -sin_a
        rot_mats[:, 1, 0] = sin_a
        rot_mats[:, 1, 1] = cos_a

        # Apply rotation in one step
        coordinates_rotated = torch.bmm(coordinates_original, rot_mats.transpose(1, 2))
        coordinates_rotated.clamp_(0.0, 1.0)

        # Scale coordinates
        scale = torch.tensor([[W, H]], device=device)
        src_points = coordinates_original * scale
        dst_points = coordinates_rotated * scale

        # Get transformation matrix and warp
        dst_points = dst_points.to(src_points.dtype)
        M = get_perspective_transform(src_points, dst_points)
        return warp_perspective(x, M, (H, W), mode="bilinear")

    def apply_transform(self, x, params, flags, transform=None):
        return self.spatial_transform_batch(x)


class ImageAugmentationPipeline:
    def __init__(self, augmentations: List[str]):
        # Generate random parameters once
        brightness_shift = random.uniform(0.0, 0.2)
        contrast_shift = random.uniform(0.0, 0.2)
        saturation_shift = random.uniform(0.0, 0.2)
        hue_shift = random.uniform(0.0, 0.2)

        # Initialize augmentations,
        # same_on_batch should always be True to keep augmentations consistent with different video frames
        self.color_jiggle = ColorJiggle(
            brightness_shift,
            contrast_shift,
            saturation_shift,
            hue_shift,
            p=0.25,
            same_on_batch=True,
        )
        self.random_planckian_jitter = K.RandomPlanckianJitter(
            select_from=[6, 12, 18, 24], p=0.25, same_on_batch=True
        )
        self.iso_noise = isoNoise(p=0.1)
        self.translation = K.RandomAffine(
            degrees=0, translate=(0.03, 0.0), p=0.25, same_on_batch=True
        )
        self.blur = Blur(blur_prob=0.2, gaussian_blur_weight=0.5)
        self.random_sharpness = K.RandomSharpness(
            sharpness=(0.5, 1.5), p=0.15, same_on_batch=True
        )
        self.random_erasing = FastRandomErasing(
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0.0,
            p=0.35,
            same_on_batch=True,
        )
        self.rrc_scale = (0.5, 1.0)
        self.rrc_ratio = (0.5, 2.0)
        self.rrc_p = 0.35
        self.random_resized_crop = FastRandomResizedCrop(
            size=(192, 192),
            scale=self.rrc_scale,
            ratio=self.rrc_ratio,
            p=self.rrc_p,
            same_on_batch=True,
            interpolation="bilinear",
        )
        aug_str_cls_map = {
            "spatial_transform": SpatialTransformBatch(),
            "color": self.color_jiggle,
            "planckian": self.random_planckian_jitter,
            "iso_noise": self.iso_noise,
            "translation": self.translation,
            "random_blur": self.blur,
            "sharpness": self.random_sharpness,
            "random_erasing": self.random_erasing,
            "random_resized_crop": self.random_resized_crop,
        }
        valid_augs = []
        for name in augmentations:
            assert name in aug_str_cls_map
            valid_augs.append(aug_str_cls_map[name])

        self.augmentations = AugmentationSequential(*valid_augs)

    def __call__(self, frames):
        return self.augmentations(frames)


class BatchRandAugment:
    """
    Wrapper around RandAugmentKornia to apply the same augmentations to a batch of videos.
    Assumes the input shape is (B, T, C, H, W).
    """

    def __init__(self, augmentations: List[str]) -> None:
        self.rand_augment = ImageAugmentationPipeline(augmentations=augmentations)

    def __call__(self, frames):
        # Set default device for tensor creation
        B, T, C, H, W = frames.shape
        frames = frames.reshape(B * T, C, H, W).float().div_(255.0)
        frames = self.rand_augment(frames)
        frames = frames.mul_(255.0).reshape(B, T, C, H, W).byte()
        return frames
