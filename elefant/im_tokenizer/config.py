from typing import Optional
import pydantic
from elefant.config import ConfigBase


class ConvTokenizerConfig(ConfigBase):
    num_tokens: int = 1


class VitTokenizerConfig(ConfigBase):
    patch_size: int = 16


class ImageTokenizerConfig(ConfigBase):
    type: str = "vit"
    conv_tokenizer_config: Optional[ConvTokenizerConfig] = pydantic.Field(
        default=ConvTokenizerConfig()
    )
    vit_tokenizer_config: Optional[VitTokenizerConfig] = pydantic.Field(
        default=VitTokenizerConfig()
    )
