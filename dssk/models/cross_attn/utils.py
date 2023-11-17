import os
import random
import torch
import numpy as np
import io
import hashlib
import json
from typing import Any, Tuple, Dict, Union, Optional
from argparse import ArgumentParser
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedModel,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from .cross_attn_gptbigcode import CrossAttnGPTBigCode


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary. Adapted from:
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
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


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds.

    Args:
        seed (int, optional): Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed as per
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def get_model(
    model_path: str,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    use_gradient_checkpointing: Optional[bool] = False,
    device: Optional[Union[str, torch.device]] = None,
    n_cross_attn_layers: Optional[int] = 0,
    cross_attn_layers_stride: Optional[int] = 1,
    cross_attn_shared_weights: Optional[bool] = True,
    cross_attn_dropout_prob: Optional[float] = 0.0,
    cross_attn_final_layer: Optional[bool] = False,
    cross_attn_shared_projections: Optional[bool] = False,
    cross_attn_hidden_size: Optional[int] = None,
    cross_attn_num_attention_heads: Optional[int] = None,
) -> PreTrainedModel:
    """Helper function to get models..
    For models that are instances of GPTBigCodeForCausalLM, optionally
    add cross-attn layers to the loaded model.

    Args:
        model_path (str): Local path or huggingface hub id to model.
        bos_token_id (int):  BOS token id to be added to model config.
        eos_token_id (int):  EOS token id to be added to model config.
        pad_token_id (int):  Padding token id to be added to model config.
        use_gradient_checkpointing (Optional[bool]): Whether to enable gradient checkpointing. Defaults to False,
        device: Optional[Union[str, torch.device, None]]:  Device where to load the model. Defaults to None.
        n_cross_attn_layers (Optional[int], optional): Number of cross-attn layers to be added to the model. Defaults to 0.
        cross_attn_layers_stride (Optional[int]): Strid for adding cross-attn layers. Defaults to 1.
        cross_attn_shared_weights (Optional[bool]): Whether to share cross-attn parameters. Deafults to True.
        cross_attn_dropout_prob (Optional[float]): Dropout probability for corss-attn attention masks. Defaults to 0.0 (no dropout.)
        cross_attn_final_layer (Optional[bool]): Whether tthe last layer is a cross-attn layer. Deafults to False.
        cross_attn_shared_projections (Optional[bool]): Whether to share parameters for query and key projections. Defaults to False.
        cross_attn_hidden_size (Optional[int]): If None (default), will use the base decoder's hiden size.
        cross_attn_num_attention_heads (Optional[int]): If None (default), will use the base decoder's number of attn heads.

    Returns:
        PreTrainedModel: Pre-trained model.
    """

    if n_cross_attn_layers > 0:
        model = CrossAttnGPTBigCode(
            model_path,
            n_cross_attn_layers=n_cross_attn_layers,
            cross_attn_layers_stride=cross_attn_layers_stride,
            cross_attn_shared_weights=cross_attn_shared_weights,
            cross_attn_dropout_prob=cross_attn_dropout_prob,
            cross_attn_final_layer=cross_attn_final_layer,
            cross_attn_shared_projections=cross_attn_shared_projections,
            cross_attn_hidden_size=cross_attn_hidden_size,
            cross_attn_num_attention_heads=cross_attn_num_attention_heads,
            randomly_initialize_decoder=False,
        )
        if device is not None:
            model = model.to(device)
        model.bos_token_id = bos_token_id
        model.eos_token_id = eos_token_id
        model = model.prepare_for_training(
            train_all_params=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        # This branch is mostly used to instantiate pre-processing models.
        # We then handle device allocation manually to avoid several processes
        # Loading models into ram simultaneously. device_map handles it if
        # device is specified.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            device_map={
                "": device or "cpu"
            },  # loads directly on device to avoid multiple processes loading models on ram.
        )

    # TODO: Why is `config` here and not above?
    model.config.pad_token_id = pad_token_id

    return model


def get_hf_training_args(
    exp_dict: Dict[str, Union[str, int, float]], args: ArgumentParser, savedir: str
) -> Seq2SeqTrainingArguments:
    """Prepare training args given exp dict and command line args.

    Args:
        exp_dict (Dict[str, Union[str, int, float]]): Experiment dict as defined in exp_configs.py
        args (ArgumentParser): parsed command line opttions.
        savedir (str): Folder where to save experiment data.

    Returns:
        Seq2SeqTrainingArguments: Training arguments.
    """
    if args.deepspeed is None or args.deepspeed.lower() == "none":
        deepspeed_config = None
    else:
        with io.open(args.deepspeed, "r", encoding="utf-8") as f:
            deepspeed_config = json.load(f)

        if "auto" not in deepspeed_config["gradient_accumulation_steps"]:
            # Override value with skip_steps from the cfg in exp_configs.py
            deepspeed_config["gradient_accumulation_steps"] = exp_dict["skip_steps"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=savedir,
        local_rank=args.local_rank,
        per_device_train_batch_size=exp_dict["train_batch_size"],
        per_device_eval_batch_size=exp_dict["test_batch_size"],
        max_steps=args.steps,
        learning_rate=exp_dict["learning_rate"],
        lr_scheduler_type=exp_dict["lr_scheduler_type"],
        warmup_ratio=exp_dict["warmup_ratio"],
        adam_beta1=exp_dict["adam_beta1"],
        adam_beta2=exp_dict["adam_beta2"],
        adam_epsilon=exp_dict["adam_epsilon"],
        weight_decay=exp_dict["weight_decay"],
        label_smoothing_factor=exp_dict["label_smoothing_factor"],
        max_grad_norm=exp_dict["max_grad_norm"],
        gradient_accumulation_steps=exp_dict["skip_steps"],
        gradient_checkpointing=exp_dict["gradient_checkpointing"],
        dataloader_num_workers=max(os.cpu_count() // args.world_size, 1),
        fp16=exp_dict["fp16"],
        bf16=exp_dict["bf16"],
        logging_dir=os.path.join(savedir, "logs"),
        logging_strategy="steps",
        logging_steps=args.log_every,
        save_strategy="steps",
        save_steps=args.log_every,
        evaluation_strategy="steps",
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_config,
        push_to_hub=args.push_to_hub,
        remove_unused_columns=False,  # This needs to be False since our custom dataset passes fields that are processed in the collator.
        log_on_each_node=False,  # Only rank 0 will write logs.
    )

    return training_args


def print_rank_0(logger, message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger(message)
    else:
        logger(message)


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
    )

    # load weights from checkpoint (might be split into multiple files)
    checkpoint = {}
    for p in glob(os.path.join(ckp_path, "pytorch_model-*.bin")):
        checkpoint |= torch.load(p, map_location=torch.device(device))

    model.load_state_dict(checkpoint)

    return model, tokenizer
