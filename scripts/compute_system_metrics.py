"""
This file provides utility code to compute runtime, flops and macs on nonsensical datasets.
"""
import math
import numpy as np
import psutil
import time
import torch

from pathlib import Path
from pprint import pprint
from typing import Optional, Any

from dssk.inference.get_interface import get_interface
from dssk.metrics.generation.system_metrics import estimate_flops_macs_params
from dssk.models import KNOWN_MODEL_TYPE, infer_model_type
from dssk.utils.scripting import get_local_rank_and_world_size

def create_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to HF model repository (local or in HF hub).",
    )

    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="Model checkpoint to be loaded with our own code.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(KNOWN_MODEL_TYPE.keys()),
        help="Specify model type. Inferred if left empty.",
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size.",
    )

    parser.add_argument(
        "--cache_path",
        default="./tmp-cache/",
        type=str,
        help="Path to cache.",
    )

    parser.add_argument(
        "--context_length",
        default=None,
        type=int,
        help="Length of input. If not provided, metrics are computed for all powers of 2 between 2 and model's max length, extreme included.",
    )

    parser.add_argument(
        "--question_length",
        default=1,
        type=int,
        help="Length of input question.",
    )

    parser.add_argument(
        "--answer_length",
        default=0,
        type=int,
        help="Length of generated answer. If unset, metrics are computed for forward mode. This script ignores eos tokens.",
    )

    parser.add_argument(
        "--has_encoder",
        action="store_true",
        default=False,
        help="Whether model has an encoder, hence context should be passed to the encoder.",
    )

    parser.add_argument(
        "--to_device",
        type=str,
        help="If a device is provided, will explicitly call `model.to(device)`.",
    )

    parser.add_argument(
        "--ds_config",
        type=str,
        help="Path to deepspeed configuration file.",
    )

    parser.add_argument(
        "--peft_config",
        type=str,
        help="Path to the PEFT fine-tuning model configuration file.",
    )

    parser.add_argument(
        "--model_peft_ckpt",
        type=str,
        help="PEFT finetuning checkpoint to be loaded.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        help="Where to save metric results.",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of times generation is run. Output runtimes will be averaged over the trials.",
    )

    return parser


def infer_default_kwargs(kwargs: dict[str, Any]) -> None:
    """Infer different kwarg defaults in a model_type-aware manner."""
    if not kwargs["model_type"]:
        kwargs["model_type"] = infer_model_type(
            model_path=kwargs["model_path"], model_ckpt=kwargs["model_ckpt"]
        )


def get_model_from_interface(kwargs):
    model_interface = get_interface(
        max_new_tokens=kwargs["answer_length"],
        model_max_length=kwargs["context_length"]
        + kwargs["question_length"]
        + kwargs["answer_length"]
        if kwargs["context_length"]
        else None,
        include_questions_on_contexts=False,
        **kwargs,
    )
    model = getattr(model_interface, "model", None)
    tokenizer = getattr(model_interface, "tokenizer", None)

    return model, tokenizer


def main(explicit_arguments: Optional[list[str]] = None) -> str:
    parser = create_parser()
    args = parser.parse_args(explicit_arguments)

    # For multiprocess. No need to set them, torchrun (or deepspeed) will handle them.
    #     world_size is the total number of workers.
    #     local_rank is the number of the present worker (in range(world_size)).
    args.local_rank, args.world_size = get_local_rank_and_world_size()

    kwargs = vars(args)
    infer_default_kwargs(kwargs)
    pprint(kwargs)

    if kwargs["save_dir"] is None:
        kwargs["save_dir"] = f"outputs/{kwargs['model_path']}"

    save_dir = Path(kwargs["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        save_dir
        / f"system-metrics-question_length={kwargs['question_length']}-answer_length={kwargs['answer_length']}.npy"
    )

    # load model, tokenizer
    model, tokenizer = get_model_from_interface(kwargs)

    # set context length values for experiments
    if kwargs["context_length"] is None:
        context_lengths = [
            2**i for i in range(1, int(math.log(tokenizer.model_max_length, 2)) + 1)
        ] + [tokenizer.model_max_length]
    else:
        context_lengths = [kwargs["context_length"]]

    # pick non-special token ids, to generate nonsensical texts
    # use different token ids at each call of generate, to make sure cache is not used
    possible_ids = set(range(len(tokenizer))) - set(
        [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
    )
    tokens = np.random.choice(list(possible_ids), len(context_lengths), replace=False).tolist()

    # prepare question tokens
    question_ids = torch.full(
        (kwargs["batch_size"], kwargs["question_length"]), tokenizer.bos_token_id
    ).to(model.device)
    question_attn_mask = torch.ones_like(question_ids).to(model.device)

    # set generation hyper-parameters
    gen_kwargs = {
        "max_new_tokens": kwargs["answer_length"],
        # Override eos_token to make sure generation is not stopped earlier
        "eos_token_id": -100,
    }

    # results and setting to be saved
    to_save = {
        "cuda_device_name": torch.cuda.get_device_name(),
        "cpu_count": psutil.cpu_count(),
        "memory": psutil.virtual_memory().total,
        "context_lengths": context_lengths,
        "tokens": tokens,
        "flops": [],
        "runtimes_avg": [],
        "runtimes_std": [],
        "macs": [],
        "params": [],
    }
    to_save |= kwargs

    for t_id, l_c in zip(tokens, context_lengths):
        # prepare context ids
        ctx_ids = torch.tile(
            torch.tensor([tokenizer.bos_token_id] + [t_id] * (l_c - 1)), (kwargs["batch_size"], 1)
        ).to(model.device)
        ctx_attn_mask = torch.ones_like(ctx_ids).to(model.device)

        # update generation arguments
        if kwargs["has_encoder"]:
            # pass context to encoder and question to decoder
            gen_kwargs |= {
                "context_ids": ctx_ids,
                "encoder_attention_mask": ctx_attn_mask,
                "input_ids": question_ids,
                "attention_mask": question_attn_mask,
            }
        else:
            # pass both to decoder
            gen_kwargs |= {
                "input_ids": torch.cat((ctx_ids, question_ids), 1),
                "attention_mask": torch.cat((ctx_attn_mask, question_attn_mask), 1),
            }

        # measure generation runtime and report avg and std
        runtimes = []
        for _ in range(kwargs["n_trials"]):
            tik = time.time()
            model.generate(**gen_kwargs)
            tok = time.time()

            runtimes.append(tok - tik)

        # ignore first run as it usually takes longer because of python related loading
        runtime_avg, runtime_std = np.average(runtimes[1:]).item(), np.std(runtimes[1:]).item()

        # measure generation flops
        all_flops, all_macs, all_params = estimate_flops_macs_params(model, gen_kwargs)

        to_save["flops"].append(all_flops)
        to_save["macs"].append(all_macs)
        to_save["params"].append(all_params)
        to_save["runtimes_avg"].append(runtime_avg)
        to_save["runtimes_std"].append(runtime_std)

        # save now in case of errors
        np.save(output_path, to_save)

        print(f"Context length {l_c}")
        print(f"Runtime {runtime_avg} +- {runtime_std}")
        print("Flops:", all_flops)

    print("Results saved in", output_path)


if __name__ == "__main__":
    main()
