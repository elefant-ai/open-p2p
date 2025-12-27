from elefant.text_tokenizer.text_tokenizer import (
    GemmaTextTokenizer,
    DummyTextTokenizer,
)
from elefant.text_tokenizer.config import (
    TextTokenizerConfig,
    DummyTextTokenizerConfig,
    GemmaTextTokenizerConfig,
)
from elefant.text_tokenizer.base_text_tokenizer import TextBaseTokenizer


def get_text_tokenizer(config: TextTokenizerConfig | None) -> TextBaseTokenizer | None:
    if not config:
        return None
    elif config.text_tokenizer_name == "gemma":
        return GemmaTextTokenizer(GemmaTextTokenizerConfig())
    elif config.text_tokenizer_name == "dummy":
        return DummyTextTokenizer(DummyTextTokenizerConfig())
    else:
        raise ValueError(f"Unknown text tokenizer: {config.text_tokenizer_name}")
