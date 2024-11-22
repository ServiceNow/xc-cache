import os
from typing import Any, Dict, Tuple
import hashlib
import json

import torch

from omegaconf import DictConfig


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary.
    Adapted from: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    Note that this assumes all keys are str.

    """
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def save_exp_dict(exp_dict: Dict[str, Any], output_path: str) -> None:
    _f_name = os.path.join(output_path, "exp_dict.json")
    if not os.path.exists(_f_name):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                with open(_f_name, "w") as f:
                    json.dump(exp_dict, f)
        else:
            with open(_f_name, "w") as f:
                json.dump(exp_dict, f)


def omegaconf_to_dict(cfg: DictConfig, types: Tuple = (int, str, bool, float)):
    """Converts a two-level nested omegaconf.DictConfig to dict, keeping values only for the chosen types.
    Useful when logging config in wandb.

    Args:
        cfg (DictConfig): config to be converted
        types (Tuple): value types to keep

    Returns:
        dict: dictionnary of {l1-key}-{l2-key}: value
    """

    config = {}
    for k1, value1 in cfg.items():
        if isinstance(value1, DictConfig):
            for k2, value2 in value1.items():
                if isinstance(value2, types):
                    config[f"{k1}-{k2}"] = value2
        else:
            if isinstance(value1, types):
                config[k1] = value1

    return config
