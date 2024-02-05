import json
import os
import torch
from safetensors.torch import load_file as load_safetensors
from dssk.models.cross_attn.get_model import get_model
from dssk.models.get_tokenizer import get_tokenizer


# Any key from that list present in config will be forwarded to get_model
CONFIG_KEYS_FORWARDED_TO_GET_MODEL = {
    "n_cross_attn_layers",
    "cross_attn_layers_stride",
    "cross_attn_shared_weights",
    "cross_attn_dropout_prob",
    "cross_attn_final_layer",
    "cross_attn_shared_projections",
    "cross_attn_hidden_size",
    "cross_attn_num_attention_heads",
    "cross_attn_num_key_value_heads",
    "cross_attn_attention_bias",
    "cross_attn_skip_connections",
    "model_type",
    "max_len",
    "include_questions_on_contexts",
}


def load_checkpoint(ckp_path, device="cpu"):
    from glob import glob

    # retrieve model config
    with open(os.path.join(ckp_path, "config.json"), "r") as f_config:
        config = json.load(f_config)

    # Identify the keys to be forwarded to get_model
    forwarded_config = {
        key: value for key, value in config.items() if key in CONFIG_KEYS_FORWARDED_TO_GET_MODEL
    }

    # load standard tokenizer
    tokenizer = get_tokenizer(config["_name_or_path"])

    # instantiate model
    model = get_model(
        model_path=config["_name_or_path"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        **forwarded_config,
    )

    # load weights from checkpoint (might be split into multiple files)
    checkpoint = {}
    # Only one of the following two `for` loops will do something.
    for p in glob(os.path.join(ckp_path, "pytorch_model-*.bin")):
        checkpoint |= torch.load(p, map_location=torch.device(device))
    for p in glob(os.path.join(ckp_path, "model-*.safetensors")):
        checkpoint |= load_safetensors(p, device=device)

    model.load_state_dict(checkpoint)

    return model, tokenizer
