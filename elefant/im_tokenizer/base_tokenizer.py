import torch
import torch.nn as nn
from abc import abstractmethod
from elefant.im_tokenizer.config import ImageTokenizerConfig


class ImageBaseTokenizer(nn.Module):
    def __init__(self, config: ImageTokenizerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_n_img_tokens(self) -> int:
        pass

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward(input_tensor)
