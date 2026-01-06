from elefant.text_tokenizer.base_text_tokenizer import TextBaseTokenizer
from elefant.text_tokenizer.config import (
    GemmaTextTokenizerConfig,
    DummyTextTokenizerConfig,
)
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Sequence, Union
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, SiglipTextModel


class DummyTextTokenizer(TextBaseTokenizer):
    # this is a dummy text tokenizer used to pass text
    def __init__(self, config: DummyTextTokenizerConfig):
        super().__init__(config)
        self.embed_dim = 2048
        self.n_text_tokens = 1

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        input_ids = torch.ones(1, self.embed_dim)
        attention_mask = torch.ones(1, self.embed_dim)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        B, text_dim = input_ids.shape
        return torch.ones(B, 1, self.n_text_tokens, self.embed_dim).to(input_ids.device)

    def get_n_text_tokens(self) -> int:
        return self.n_text_tokens

    def get_text_embed_dim(self) -> int:
        return self.embed_dim


class GemmaTextTokenizer(TextBaseTokenizer):
    def __init__(self, config: GemmaTextTokenizerConfig):
        super().__init__(config)
        self.gemma_model = SentenceTransformer(config.model_id).eval()
        self.tokenizer = self.gemma_model.tokenizer
        self.gemma_embedding = self.gemma_model[0].auto_model.eval()
        self.embed_dim = self.gemma_model.get_sentence_embedding_dimension()
        del self.gemma_model
        self.n_text_tokens = 1
        self.max_position_embeddings = config.max_position_embeddings

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        return self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_position_embeddings,
        )

    @torch.compiler.disable
    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if len(input_ids.shape) == 3:
            # batch process for training
            B, T, text_dim = input_ids.shape
        elif len(input_ids.shape) == 2:
            # inference time
            B, text_dim = input_ids.shape
            T = 1
        else:
            raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")
        input_ids = input_ids.reshape(-1, text_dim).to(self.gemma_embedding.device)
        attention_mask = attention_mask.reshape(-1, text_dim).to(
            self.gemma_embedding.device
        )
        text_features = self.gemma_embedding(
            input_ids, attention_mask
        ).last_hidden_state
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(text_features.size()).float()
        )
        sum_embeddings = torch.sum(text_features * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask

        sentence_embedding = sentence_embedding.reshape(
            B, T, self.n_text_tokens, self.embed_dim
        )
        return sentence_embedding

    def get_n_text_tokens(self) -> int:
        return self.n_text_tokens

    def get_text_embed_dim(self) -> int:
        return self.embed_dim
