# Copyright 2024 ServiceNow
# This file contains code by the authors denoted below, which has been modified from its original version.
#
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import logging
import argparse
from typing import Optional
from pathlib import Path
import torch

from xc_cache.utils.scripting import print_rank_0, get_local_rank_and_world_size
from xc_cache.utils.scripting import parse_bool_flag, set_random_seed
from xc_cache.utils.jobs import save_exp_dict
from xc_cache.utils.jobs import dict_hash
from xc_cache.models.peft.get_model import get_model, get_cross_model
from xc_cache.models.get_tokenizer import get_tokenizer
from scripts.peft.config.exp_configs import EXP_GROUPS
from xc_cache.data.peft.datasets_loader import xc_cache_data_prep
from xc_cache.train.peft.trainer import get_trainer
from xc_cache.train.peft.get_training_args import get_training_args
from xc_cache.utils.hf import get_model_path_from_config

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# This function has been adapted from code in transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
def toto(self, seq_len, device, dtype):
    seq_len = 8192
    self.max_seq_len_cached = seq_len
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

    freqs = torch.outer(t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


LlamaRotaryEmbedding._set_cos_sin_cache = toto

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


set_random_seed()

RESULTS_FNAME = "results.ipynb"


def parse_training_args(args: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        help="Experiment id.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        required=True,
        help="Define the base directory where data will be cached.",
    )
    parser.add_argument(
        "--dataset",
        default="ServiceNow/xc_cache_training_data",
    )
    parser.add_argument(
        "--cache_dir",
        default="./tmp-cache/",
        help="Define the base directory or cache may be written by hf.",
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument("-j", "--job_scheduler", default=None, help="Choose Job Scheduler.")
    parser.add_argument("--python_binary", default="python", help="path to your python executable")
    parser.add_argument("--steps", default=500_000, type=int, help="Number of training steps.")
    parser.add_argument(
        "--eval_samples",
        default=1000,
        type=int,
        help="Number of samples to use from the validation sets to compute the validation loss.",
    )
    parser.add_argument(
        "--do_extra_evals",
        type=parse_bool_flag,
        default="false",
        help="Whether to run Q&A specific evaluations that require generation.",
    )
    parser.add_argument(
        "--generation_eval_max_sample_size",
        type=int,
        default=100,
        help="Number of samples to use for extra evaluationsthat require generation.",
    )
    parser.add_argument(
        "--log_every",
        default=1000,
        type=int,
        help="Number of iterations to wait before logging training scores.",
    )
    parser.add_argument(
        "--wandb_entity_name",
        type=str,
        default=None,
        help="Name of wandb entity for reporting.",
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default=None, help="Name of wandb project."
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Name of run.")
    parser.add_argument(
        "--wandb_log_gradients",
        type=parse_bool_flag,
        default="false",
        help="Whether to write gradients to wandb logs.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=parse_bool_flag,
        default="false",
        help="Whether to push model to hf hub.",
    )
    parser.add_argument(
        "--deepspeed",
        default=None,
        type=str,
        help="""Optional path to deepspeed config.""",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )

    return parser.parse_args(args)


def train(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    try:
        savedir = os.path.join(savedir, args.exp_id)
    except TypeError:
        exp_id = dict_hash(exp_dict)
        savedir = os.path.join(savedir, exp_id)
        print(savedir)
        if os.path.exists(savedir):
            logger.warning(
                "A folder for this config was created in the past. Will resume training if ckpts are available."
            )

    # Ensure that savedir exists
    Path(savedir).mkdir(parents=True, exist_ok=True)
    # This will save exp_dict in savedir if it's not already there.
    save_exp_dict(exp_dict, savedir)

    if "checkpoint_path" in exp_dict:
        assert "model_path" not in exp_dict or not exp_dict["model_path"]

        decoder = get_cross_model(
            exp_dict["checkpoint_path"],
            exp_dict["lora_cfg"],
            exp_dict["load_in_8bit"],
        )
        model_path = get_model_path_from_config(
            os.path.join(exp_dict["checkpoint_path"], "config.json")
        )
        tokenizer = get_tokenizer(model_path, add_eos_token=True)
    else:
        decoder = get_model(
            model_path=exp_dict["model_path"],
            lora_config=exp_dict["lora_cfg"],
            cache_dir=args.cache_dir,
            load_in_8bit=exp_dict["load_in_8bit"],
            process_index=int(os.environ.get("LOCAL_RANK", 0)),
        )
        tokenizer = get_tokenizer(exp_dict["model_path"], add_eos_token=True)

    # If kept True, this lead the model to wrongly expect
    # tensors of the wrong shape.
    decoder.config.use_cache = False

    training_data, validation_data = xc_cache_data_prep(
        tokenizer=tokenizer,
        context_length=exp_dict["context_size"],
        dataset_name=args.dataset,
        num_val_samples=args.eval_samples,
        model_type=exp_dict["model_type"],
        cross_format=("checkpoint_path" in exp_dict),
        data_cache_dir=args.cache_dir,
    )

    training_args = get_training_args(exp_dict, args, savedir)

    print_rank_0(logger.info, training_args)
    print_rank_0(logger.info, decoder)
    print_rank_0(logger.info, training_data)
    print_rank_0(logger.info, validation_data)

    wandb_run_name = args.wandb_run_name

    trainer = get_trainer(
        model=decoder,
        tokenizer=tokenizer,
        args=training_args,
        training_data=training_data,
        validation_data=validation_data,
        maximum_input_length=exp_dict["context_size"],
        wandb_entity_name=args.wandb_entity_name,
        wandb_project_name=args.wandb_project_name,
        wandb_run_name=wandb_run_name,
        wandb_log_grads=args.wandb_log_gradients,
        cross_trainer=("checkpoint_path" in exp_dict),
        cross_trainer_do_extra_evals=args.do_extra_evals,
        cross_trainer_generation_eval_max_sample_size=args.generation_eval_max_sample_size,
    )

    trainer.train(
        resume_from_checkpoint=any(dir.startswith("checkpoint") for dir in os.listdir(savedir))
    )

    logger.info("Experiment done\n")


def main(explicit_arguments: Optional[list[str]] = None) -> None:
    args = parse_training_args(explicit_arguments)
    args.local_rank, args.world_size = get_local_rank_and_world_size()

    _NGPUS = torch.cuda.device_count()

    args.python_binary = args.python_binary.replace("NGPUS", str(_NGPUS))

    print_rank_0(logger.info, args)

    # get experiment configurations list
    exp_configs = EXP_GROUPS[args.exp_group]
    # add number of steps to exp dict since it influences the learning rate scheduler
    # and could be then seen as a hyperparameter.
    updated_exp_configs = [{**cfg, **{"steps": args.steps}} for cfg in exp_configs]

    train(
        updated_exp_configs[0],
        os.path.join(args.savedir_base),
        args,
    )
