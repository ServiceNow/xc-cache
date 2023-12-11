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
import os

from pathlib import Path
from datasets import load_dataset

from baselines.fid.src.datasets_loader import Collator
from baselines.fid.src.t5_wrapper import FiDT5
from baselines.fid.src.save_load_model import load, set_optim
from baselines.fid.trainer import train
from baselines.fid.options import Options


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
    Handle single and multi-GPU / multi-node.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    has_local_rank = hasattr(params, "local_rank")

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    if has_local_rank and params.local_rank != -1:
        assert params.main_port == -1

        # read environment variables
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["NGPU"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.is_distributed = True

    else:
        n_gpu = torch.cuda.device_count()
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = n_gpu
        params.n_gpu_per_node = n_gpu
        params.is_distributed = False

    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

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


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    init_distributed_mode(opt)

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = init_logger(opt.is_main, opt.is_distributed, checkpoint_path / "run.log")

    model_name = "t5-" + opt.model_size

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_name, model_max_length=opt.text_maxlength
    )
    collator = Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use global rank and world size to split the eval set on multiple gpus
    train_dataset = load_dataset(
        f"ServiceNow/{opt.dataset_name}", cache_dir=opt.cache_path, split="train"
    )
    eval_dataset = load_dataset(
        f"ServiceNow/{opt.dataset_name}", cache_dir=opt.cache_path, split="val"
    )

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / "checkpoint" / "latest"
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = load(
            FiDT5, load_path, opt, logger, reset_params=False
        )
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = load(
            FiDT5, opt.model_path, opt, logger, reset_params=True
        )
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

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
    )
