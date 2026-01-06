import torch
import torch.nn as nn
import torchvision
from elefant.im_tokenizer.config import ImageTokenizerConfig
from elefant.im_tokenizer.base_tokenizer import ImageBaseTokenizer
from elefant.torch import eager_assert


class ConvTokenizer(ImageBaseTokenizer):
    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)
        self.num_tokens = config.conv_tokenizer_config.num_tokens

        # Loosely following https://arxiv.org/abs/2104.04258
        # We start from the pretrained efficientnet model but allow the weights to be trained.
        efficientnet = torchvision.models.efficientnet_b0(
            dropout=0.2, weights="IMAGENET1K_V1"
        )

        # TODO: could try 0:7
        self.efficientnet_preprocess = efficientnet.features[0:6]

        if frame_height == 192 and frame_width == 192:
            self.n_output_features = 112 * 12 * 12
            self.n_sections = 12
        elif frame_height == 256 and frame_width == 256:
            self.n_output_features = 112 * 16 * 16
            self.n_sections = 16
        else:
            raise ValueError(f"Unsupported frame size: {frame_height}x{frame_width}")

        self.mlp = nn.Sequential(
            nn.Linear(self.n_output_features, embed_dim * self.num_tokens),
            nn.LayerNorm(embed_dim * self.num_tokens),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        features = self.efficientnet_preprocess(x.reshape(B * T, C, H, W)).contiguous()
        eager_assert(features.shape, (B * T, 112, self.n_sections, self.n_sections))
        features = features.reshape(T * B, self.n_output_features)
        y = self.mlp(features)
        y = y.reshape(B, T, self.get_n_img_tokens(), self.embed_dim)
        return y

    def get_n_img_tokens(self) -> int:
        return self.num_tokens
