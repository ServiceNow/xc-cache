from typing import Tuple
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast


def get_tokenizer(
    model_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Helper function to get tokenizer and add missing special tokens.

    Args:
        model_path (str): Local path or huggingface hub id.

    Returns:
        PreTrainedTokenizerFast: Pre-trained tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding="max_length", truncation="max_length"
    )

    pad_token = tokenizer.pad_token
    if pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    return tokenizer
