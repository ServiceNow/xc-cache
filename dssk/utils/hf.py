import json
from typing import Dict, Any


def get_model_path_from_config(config_path: str) -> Dict[str, Any]:

    with open(config_path, "rt") as fp:
        model_config = json.load(fp)

    return model_config["_name_or_path"]
