from elefant.im_tokenizer.config import ImageTokenizerConfig
from elefant.im_tokenizer.base_tokenizer import ImageBaseTokenizer
from elefant.im_tokenizer import conv_tokenizer
from elefant.im_tokenizer.tokenizer import (
    VitImageTokenizer,
    DinoV2Tokenizer,
)


def get_tokenizer(
    config: ImageTokenizerConfig, embed_dim: int, frame_height: int, frame_width: int
) -> ImageBaseTokenizer:
    if config.type == "vit":
        return VitImageTokenizer(config, frame_height, frame_width, embed_dim)
    elif config.type == "conv":
        # Note: IdentityTokenizer was removed as it wasn't used here previously.
        # If needed, import it from elefant.im_tokenizer.tokenizer
        return conv_tokenizer.ConvTokenizer(
            config, frame_height, frame_width, embed_dim
        )
    elif config.type == "dinov2":
        return DinoV2Tokenizer(config, frame_height, frame_width, embed_dim)
    else:
        raise ValueError(f"Unknown tokenizer: {config.type}")
