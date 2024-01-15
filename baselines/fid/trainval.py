# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import logging
import sys
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


def main(explicit_arguments: Optional[list[str]] = None) -> str:
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    options.add_eval_options()
    opt = options.parse(explicit_arguments)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if opt.debug else logging.INFO,
    )
    logger = logging.getLogger(__name__)

    opt.local_rank, opt.world_size = get_local_rank_and_world_size()

    set_random_seed(
        opt.local_rank + opt.seed
    )  # different seed for different sampling depending on global_rank

    model_name = "t5-" + opt.model_size
    # maximal number of tokens for T5 is 512
    text_maxlentgh = min(opt.text_maxlength, 512)
    checkpoint_path = Path(opt.name) / model_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

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
    # keep only datasets chosen for validation
    eval_dataset = eval_dataset.filter(
        lambda example: example["dataset"] in opt.val_datasets.split(";")
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
        resume_ckpt = len(list(checkpoint_path.glob("checkpoint*"))) > 0

    # train and evaluate on validation set + logging and checkpointing
    trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("Experiment done\n")


if __name__ == "__main__":
    main()
