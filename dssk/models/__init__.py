import os
import json
from typing import Optional

__all__ = ["KNOWN_MODEL_TYPE", "get_model_string", "infer_model_type"]

# Associate model types with substrings that hint that this type may apply.
KNOWN_MODEL_TYPE = {
    "fid": {"fid"},
    "llama": {"llama", "tulu"},
    "mistral": {"mistral"},
    "gptbigcode": {"code"},
    "toto": set(),
}


def get_model_string(model_path: Optional[str] = None, model_ckpt: Optional[str] = None) -> str:
    """Validate model_path and model_ckpt, and return the one that is non-None

    Exactly one of them must be non-none: that is the "model_string".
    """
    if model_path is None and model_ckpt is None:
        raise ValueError("One of model_path or model_ckpt must be specified.")
    if model_path is not None and model_ckpt is not None:
        raise ValueError("Only one of model_path or model_ckpt may be specified.")

    if model_ckpt is None:
        return model_path
    else:
        # Check for model path within config.
        try:
            with open(os.path.join(model_ckpt, "config.json")) as f:
                cfg = json.load(f)

            return cfg["_name_or_path"]
        except (FileNotFoundError, KeyError):
            return model_ckpt


def infer_model_type(model_path: Optional[str] = None, model_ckpt: Optional[str] = None) -> str:
    model_string = get_model_string(model_path=model_path, model_ckpt=model_ckpt).lower()
    candidates = set()
    for model_type, substrings in KNOWN_MODEL_TYPE.items():
        for substring in substrings:
            if substring in model_string:
                candidates.add(model_type)
                continue
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates:
        raise ValueError("Cannot infer model_type: no candidate found.")
    else:
        raise ValueError("Cannot infer model_type: too many candidates (ambiguous).")
