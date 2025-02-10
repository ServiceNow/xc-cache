import argparse
import sys
import os
from pathlib import Path
import logging
from typing import Optional
import torch
from xc_cache.utils.scripting import print_rank_0, get_local_rank_and_world_size
from xc_cache.utils.scripting import parse_bool_flag, set_random_seed
from xc_cache.utils.jobs import save_exp_dict
from xc_cache.utils.jobs import dict_hash
from xc_cache.models.cross_attn.get_model import get_model
from xc_cache.models.get_tokenizer import get_tokenizer
from scripts.cross_attn.config.exp_configs import EXP_GROUPS
from xc_cache.data.cross_attn.datasets_loader import data_prep
from xc_cache.train.cross_attn.trainer import get_trainer
from xc_cache.train.cross_attn.get_training_args import get_training_args

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
        required=True,
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
        "--data_path",
        required=True,
        help="Define the base directory or the id where training data will be found.",
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
        help="Define the base directory or the id where training data will be found.",
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

    tokenizer = get_tokenizer(exp_dict["model_path"], add_eos_token=True)

    training_data, validation_data = data_prep(
        tokenizer_path=exp_dict["model_path"],  # Used to get the right tokenizer.
        data_dir=args.data_path,
        context_length=exp_dict["context_size"],
        chunked_contexts=exp_dict["chunked_contexts"],
        training_data_subset=exp_dict["training_data_subset"],
        validation_data_subset=exp_dict["validation_data_subset"],
        data_cache_dir=args.cache_dir,
        include_context_ids=exp_dict["include_context_ids"],
        include_questions_on_contexts=exp_dict["include_questions_on_contexts"],
        model_type=exp_dict["model_type"],
    )

    decoder = get_model(
        exp_dict["model_path"],
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        use_gradient_checkpointing=exp_dict["gradient_checkpointing"],
        n_cross_attn_layers=exp_dict["num_cross_attn_layers"],
        cross_attn_layers_stride=exp_dict["cross_attn_layers_stride"],
        cross_attn_shared_weights=exp_dict["cross_attn_shared_weights"],
        cross_attn_dropout_prob=exp_dict["cross_attn_dropout_prob"],
        cross_attn_final_layer=exp_dict["cross_attn_final_layer"],
        cross_attn_shared_projections=exp_dict["cross_attn_shared_projections"],
        cross_attn_hidden_size=exp_dict["cross_attn_hidden_size"],
        cross_attn_num_attention_heads=exp_dict["cross_attn_num_attention_heads"],
        cross_attn_num_key_value_heads=exp_dict["cross_attn_num_key_value_heads"],
        cross_attn_attention_bias=exp_dict["cross_attn_attention_bias"],
        cross_attn_skip_connections=exp_dict["cross_attn_skip_connections"],
        model_type=exp_dict["model_type"],
        max_len=exp_dict["context_size"],
        include_questions_on_contexts=exp_dict["include_questions_on_contexts"],
        chunked_contexts=exp_dict["chunked_contexts"],
        cache_dir=args.cache_dir,
    )

    training_args = get_training_args(exp_dict, args, savedir)

    print_rank_0(logger.info, training_args)
    print_rank_0(logger.info, decoder)
    print_rank_0(logger.info, f"Training data size: {len(training_data)}")
    print_rank_0(logger.info, f"Validation data size: {len(validation_data)}")

    total_params = 0
    trainable_params = 0

    for _, v in decoder.named_parameters():
        total_params += v.data.numel()
        if v.requires_grad:
            trainable_params += v.data.numel()

    print_rank_0(
        logger.info, f"Total parameters: {total_params}\nTrainable parameters: {trainable_params}"
    )

    trainer = get_trainer(
        model=decoder,
        tokenizer=tokenizer,
        maximum_input_length=exp_dict["context_size"],
        args=training_args,
        training_data=training_data,
        validation_data=validation_data,
        wandb_entity_name=args.wandb_entity_name,
        wandb_project_name=args.wandb_project_name,
        wandb_run_name=args.wandb_run_name,
        wandb_log_grads=args.wandb_log_gradients,
        do_extra_evals=args.do_extra_evals,
        generation_eval_max_sample_size=args.generation_eval_max_sample_size,
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

    # Only the first experiment in exp_configs.EXP_GROUPS[args.exp_group] will run.
    train(
        updated_exp_configs[0],
        args.savedir_base,
        args,
    )


if __name__ == "__main__":
    main()
