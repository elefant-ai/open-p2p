from typing import Optional, List
import pydantic
from elefant.config import ConfigBase


class GemmaTextTokenizerConfig(ConfigBase):
    model_id: str = "google/embeddinggemma-300M"
    max_position_embeddings: int = 128


class DummyTextTokenizerConfig(ConfigBase):
    model_id: str = "dummy"


class TextTokenizerConfig(ConfigBase):
    text_tokenizer_name: Optional[str] = None
    text_annotation_model_version: List[str] = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-thinking-0905",
    ]
    # [token, dimension], set default dim to 1
    # so that the mlp won't take much memory
    text_embedding_shape: List[int] = [1, 1]
