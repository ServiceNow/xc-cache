import json
import os
import torch
from dssk.models.cross_attn.get_model import get_model
from dssk.models.get_tokenizer import get_tokenizer


def load_checkpoint(ckp_path, device="cpu"):
    from glob import glob

    # retrieve model config
    with open(os.path.join(ckp_path, "config.json"), "r") as f_config:
        config = json.load(f_config)

    # load standard tokenizer
    tokenizer = get_tokenizer(config["_name_or_path"])

    # instantiate model
    model = get_model(
        model_path=config["_name_or_path"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        n_cross_attn_layers=config["n_cross_attn_layers"],
        cross_attn_layers_stride=config["cross_attn_layers_stride"],
        cross_attn_shared_weights=config["cross_attn_shared_weights"],
        cross_attn_dropout_prob=config["cross_attn_dropout_prob"],
        cross_attn_final_layer=config["cross_attn_final_layer"],
        cross_attn_shared_projections=config["cross_attn_shared_projections"],
        cross_attn_hidden_size=config["cross_attn_hidden_size"],
        cross_attn_num_attention_heads=config["cross_attn_num_attention_heads"],
        cross_attn_num_key_value_heads=config["cross_attn_num_key_value_heads"],
        cross_attn_attention_bias=config["cross_attn_attention_bias"],
        is_llama=config["is_llama"],
    )

    # load weights from checkpoint (might be split into multiple files)
    checkpoint = {}
    for p in glob(os.path.join(ckp_path, "pytorch_model-*.bin")):
        checkpoint |= torch.load(p, map_location=torch.device(device))

    model.load_state_dict(checkpoint)

    return model, tokenizer
