from transformers import AutoTokenizer, PreTrainedTokenizerFast


def get_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """Helper function to get tokenizer and add missing special tokens.

    Args:
        model_path (str): Local path or huggingface hub id.

    Returns:
        PreTrainedTokenizerFast: Pre-trained tokenizer.

    TODO: Handle model_max_length.
    TODO: Ensure that only scripts call this.
    TODO: Refactor this to a dssk/getters.py, where all get_something will reside.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding="max_length", truncation="max_length"
    )

    pad_token = tokenizer.pad_token
    if pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.bos_token})

    return tokenizer
