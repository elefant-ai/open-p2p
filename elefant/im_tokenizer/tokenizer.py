import logging
import os
from abc import abstractmethod
import torch
import torch.nn as nn
from torch.nn import functional as F
from elefant.im_tokenizer.config import ImageTokenizerConfig
from huggingface_hub import login, snapshot_download
from elefant.torch import eager_assert
from elefant.torch import eager_assert
from elefant.modules import LayerNormF32
from elefant.im_tokenizer import conv_tokenizer
from elefant.im_tokenizer.base_tokenizer import ImageBaseTokenizer
from typing import Tuple


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the images of shape [B, T, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, T, C, H, W = x.shape
    x = x.reshape(B, T, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 3, 5, 2, 4, 6)  # [B, T, H', W', C, p_H, p_W]
    x = x.flatten(2, 3)  # [B, T, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(3, 5)  # [B, T H'*W', C*p_H*p_W]
    return x


class IdentityTokenizer(ImageBaseTokenizer):
    """
    This tokenizer is used when we don't want to tokenize the image just pass it through.
    Only useful for unit testing.
    """

    def __init__(self, config: ImageTokenizerConfig, n_img_tokens: int):
        super().__init__(config)
        self.n_img_tokens = n_img_tokens

    def get_n_img_tokens(self) -> int:
        return self.n_img_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B, T, C * H, W)
        assert x.shape[2] == self.n_img_tokens
        return x


class VitImageTokenizer(ImageBaseTokenizer):
    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)
        self.n_img_tokens = (frame_height // config.vit_tokenizer_config.patch_size) * (
            frame_width // config.vit_tokenizer_config.patch_size
        )
        self.patch_size = config.vit_tokenizer_config.patch_size
        self.proj_to_embed_dim = nn.Linear(
            self.patch_size * self.patch_size * 3, embed_dim
        )
        # self.post_norm = LayerNormF32(embed_dim)

    def get_n_img_tokens(self) -> int:
        return self.n_img_tokens

    def forward(self, img: torch.Tensor):
        patches = img_to_patch(img, self.patch_size)
        patches = self.proj_to_embed_dim(patches)
        # TODO: decide if this is good or not.
        # patches = self.post_norm(patches)
        return patches


class DinoV2Tokenizer(ImageBaseTokenizer):
    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.eval()
        # Set model parameters to not trainable.
        for param in self.model.parameters():
            param.requires_grad = False

        if frame_height != 192 or frame_width != 192:
            raise ValueError("DinoV2Tokenizer only supports 192x192 images")
        self.n_image_tokens = 196

        self.embed_dim = embed_dim
        self.proj_to_embed_dim = nn.Linear(768, embed_dim)

    def get_n_img_tokens(self) -> int:
        return self.n_image_tokens

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = input_tensor.shape
        with torch.no_grad():
            # Pad the input tensor to 196.
            x = F.pad(input_tensor, (2, 2, 2, 2), value=0.0)
            features = self.model.forward_features(x.view(B * T, C, 196, 196))

        features = self.proj_to_embed_dim(features["x_norm_patchtokens"])
        eager_assert(features.shape, (B * T, self.n_image_tokens, self.embed_dim))
        features = features.view(B, T, self.n_image_tokens, self.embed_dim)
        return features
