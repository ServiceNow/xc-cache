from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def get_tokenizer(
    model_path: str, add_eos_token: Optional[bool] = None
) -> PreTrainedTokenizerFast:
    """Helper function to get tokenizer and add missing special tokens.

    Args:
        model_path (str): Local path or huggingface hub id.
        add_eos_token (Optional[bool]): Whether the tokenizer should append the eos token.

    Returns:
        PreTrainedTokenizerFast: Pre-trained tokenizer.

    TODO: Handle model_max_length.
    TODO: Ensure that only scripts call this.
    TODO: Refactor this to a xc_cache/getters.py, where all get_something will reside.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding="max_length", truncation="max_length", padding_side="left"
    )

    pad_token = tokenizer.pad_token
    if pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.bos_token})

    if add_eos_token is not None:
        tokenizer.add_eos_token = add_eos_token

    return tokenizer
