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

from datasets import load_dataset
from pathlib import Path
from typing import Optional

from baselines.fid.src.datasets_loader import Collator
from baselines.fid.src.t5_wrapper import FiDT5
from baselines.fid.get_training_args import get_training_args
from baselines.fid.trainer import get_trainer
from baselines.fid.options import Options
from dssk.utils.scripting import get_local_rank_and_world_size, set_random_seed, print_rank_0

from torch.distributed.elastic.multiprocessing.errors import record


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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    init_distributed_mode(opt)
    set_random_seed(
        opt.local_rank + opt.seed
    )  # different seed for different sampling depending on global_rank

    model_name = "t5-" + opt.model_size
    # maximal number of tokens for T5 is 512
    text_maxlentgh = min(opt.text_maxlength, 512)
    checkpoint_path = Path(opt.name) / model_name

    model = FiDT5.from_pretrained(model_name)
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_name, model_max_length=text_maxlentgh
    )
    collator = Collator(
        text_maxlentgh,
        tokenizer,
        answer_maxlength=opt.answer_maxlength,
        max_contexts=opt.max_contexts,
    )

    train_dataset = load_dataset(
        f"ServiceNow/{opt.dataset_name}", cache_dir=opt.cache_path, split="train"
    )
    eval_dataset = load_dataset(
        f"ServiceNow/{opt.dataset_name}", cache_dir=opt.cache_path, split="val"
    )

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    # instantiate HF dataobject for trainer's configuration
    training_args = get_training_args(opt, checkpoint_path, opt.local_rank, opt.world_size)

    # print model and other stats
    if opt.debug:
        print_rank_0(logger.info, training_args)
        print_rank_0(logger.info, model)
        print_rank_0(logger.info, f"Training data size: {len(train_dataset)}")
        print_rank_0(logger.info, f"Validation data size: {len(eval_dataset)}")

        total_params = 0
        trainable_params = 0

        for _, v in model.named_parameters():
            total_params += v.data.numel()
            if v.requires_grad:
                trainable_params += v.data.numel()

        print_rank_0(
            logger.info,
            f"Total parameters: {total_params}\nTrainable parameters: {trainable_params}",
        )

    # get HF trainer
    trainer = get_trainer(
        model=model,
        data_collator=collator,
        opt=opt,
        training_args=training_args,
        training_data=train_dataset,
        validation_data=eval_dataset,
    )
    # resume training from specified checkpoint path
    if opt.model_path is not None:
        resume_ckpt = opt.model_path
    # or from the last checkpoint saved in savedir. If no checkpoints are found, training is done from scratch
    else:
        resume_ckpt = any(dir.startswith("checkpoint") for dir in checkpoint_path.iterdir())

    # train and evaluate on validation set + logging and checkpointing
    trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("Experiment done\n")


if __name__ == "__main__":
    main()
