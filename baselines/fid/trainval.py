# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import logging
import sys
import torch
import transformers
import wandb

from datasets import load_dataset
from pathlib import Path
from typing import Optional

from baselines.fid.src.datasets_loader import Collator
from baselines.fid.src.t5_wrapper import FiDT5
from baselines.fid.src.save_load_model import load, set_optim
from baselines.fid.trainer import train
from baselines.fid.options import Options
from dssk.utils.scripting import get_local_rank_and_world_size

from torch.distributed.elastic.multiprocessing.errors import record


def init_logger(is_main=True, is_distributed=False, filename=None):
    logger = logging.getLogger()

    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return logger


def init_distributed_mode(params):
    """
    Handle single and multi-GPU.
    Initialize the following variables:
        - local_rank
        - world_size
    """

    params.local_rank, params.world_size = get_local_rank_and_world_size()
    params.is_distributed = params.world_size > 1
    params.is_main = params.local_rank == 0

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device

    # initialize multi-GPU
    if params.is_distributed:
        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        # print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


@record
def main(explicit_arguments: Optional[list[str]] = None) -> str:
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse(explicit_arguments)

    torch.manual_seed(opt.seed)
    init_distributed_mode(opt)

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name

    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = init_logger(opt.is_main, opt.is_distributed, checkpoint_path / "run.log")

    model_name = "t5-" + opt.model_size

    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_name, model_max_length=opt.text_maxlength
    )
    # maximal number of tokens for T5 is 512
    collator = Collator(
        min(opt.text_maxlength, 512), tokenizer, answer_maxlength=opt.answer_maxlength
    )

    train_dataset = load_dataset(
        f"ServiceNow/{opt.dataset_name}", cache_dir=opt.cache_path, split="train"
    )
    eval_dataset = load_dataset(
        f"ServiceNow/{opt.dataset_name}", cache_dir=opt.cache_path, split="val"
    )
    if opt.model_path != "none":  # either load specific checkpoint
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = load(
            FiDT5, opt.model_path, opt, logger, reset_params=True
        )
        logger.info(f"Model loaded from {opt.model_path}")
    elif (
        len(list(checkpoint_path.glob("checkpoint/step*"))) > 0
    ):  # or load latest checkpoint if found
        latest_path = checkpoint_path / "checkpoint" / "latest"

        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = load(
            FiDT5, latest_path.readlink(), opt, logger, reset_params=False
        )
        logger.info(f"Model loaded from {latest_path}")
    else:  # or instantiate new model
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = set_optim(opt, model)
        step, best_dev_em = 0, 0.0

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    wandb_run = wandb.init(
        name=None,
        project=opt.name,
        mode="disabled" if opt.debug or opt.local_rank > 0 else None,  # no logging while debugging
    )
    wandb_run.log(vars(opt))

    logger.info("Start training")
    train(
        model,
        tokenizer,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path,
        logger,
        wandb_run,
    )


if __name__ == "__main__":
    main()
