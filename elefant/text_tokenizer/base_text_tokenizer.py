import torch
import torch.nn as nn
from abc import abstractmethod
from elefant.text_tokenizer.config import TextTokenizerConfig


class TextBaseTokenizer(nn.Module):
    def __init__(self, config: TextTokenizerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def get_text_embed_dim(self) -> int:
        pass

    @abstractmethod
    def get_n_text_tokens(self) -> int:
        pass

    @abstractmethod
    def get_max_position_embeddings(self) -> int:
        return self.config.max_position_embeddings

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)
