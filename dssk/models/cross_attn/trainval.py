import sys
import os
from pathlib import Path
import logging
import torch
import resources.exp_configs as exp_configs
from dssk.models.cross_attn.datasets_loader import nq_prep
from dssk.models.cross_attn.hf_trainer import get_trainer
from dssk.models.cross_attn.utils import (
    get_model,
    get_tokenizer,
    get_hf_training_args,
    print_rank_0,
    set_random_seed,
    dict_hash,
    save_exp_dict,
)
from dssk.models.cross_attn.parse_training_args import parse_training_args
from dssk.models.cross_attn.constants import RESULTS_FNAME

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

set_random_seed()


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

    tokenizer = get_tokenizer(
        exp_dict["model_path"],
    )

    training_data, validation_data = nq_prep(
        exp_dict["model_path"],  # Used to get the right tokenizer.
        args.data_path,
        exp_dict["context_size"],
        exp_dict["do_repetition_augmentations"],
        exp_dict["include_context_ids"],
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
    )

    training_args = get_hf_training_args(exp_dict, args, savedir)

    print_rank_0(logger.info, training_args)
    print_rank_0(logger.info, decoder)
    print_rank_0(logger.info, f"Training data size: {len(training_data)}")
    print_rank_0(logger.info, f"Validation data size: {len(validation_data)}")

    total_params = 0
    trainable_params = 0

    for _, v in decoder.named_parameters():
        total_params += v.data.numel()
        if v.requires_grad_:
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
    )

    trainer.train(
        resume_from_checkpoint=any(dir.startswith("checkpoint") for dir in os.listdir(savedir))
    )

    logger.info("Experiment done\n")


def main(args, others):
    try:
        # LOCAL_RANK and WORLD_SIZE should be properly set automatically if
        # the script was launched with torchrun.
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        # This should be the case when a single-node single-gpu experiment is run.
        args.local_rank = 0
        args.world_size = 1

    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler == "toolkit":
        from resources.job_configs import JOB_CONFIG

        job_config = JOB_CONFIG[args.exp_group]
        _NGPUS = job_config["resources"]["gpu"]
    else:
        _NGPUS = torch.cuda.device_count()

    args.python_binary = args.python_binary.replace("NGPUS", str(_NGPUS))

    print_rank_0(logger.info, args)
    print_rank_0(logger.info, others)

    if args.job_scheduler == "toolkit":
        try:
            from haven import haven_wizard as hw
        except ImportError:
            logger.error("Submitting to toolkit requires haven-ai to be installed.")
        # Run experiments and create results file.
        # One job will be submitted to toolkit for each config dict in exp_configs.EXP_GROUPS[args.exp_group].
        hw.run_wizard(
            func=train,
            exp_list=exp_configs.EXP_GROUPS[args.exp_group],
            savedir_base=args.savedir_base,
            reset=args.reset,
            job_config=job_config,
            results_fname=RESULTS_FNAME,
            python_binary_path=args.python_binary,
            args=args,
        )
    else:
        # For local experiments, not submitted to a scheduler,
        # Only the first experiment in exp_configs.EXP_GROUPS[args.exp_group] will run.
        train(
            exp_configs.EXP_GROUPS[args.exp_group][0],
            args.savedir_base,
            args,
        )


if __name__ == "__main__":
    args, others = parse_training_args()
    main(args, others)
