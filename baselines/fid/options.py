# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import argparse
import logging

logger = logging.getLogger()


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument("--warmup_steps", type=int, default=1000)
        self.parser.add_argument("--total_steps", type=int, default=1000)
        self.parser.add_argument(
            "--scheduler_steps",
            type=int,
            default=None,
            help="total number of steps for the scheduler, if None then scheduler_total_step = total_step",
        )
        self.parser.add_argument("--accumulation_steps", type=int, default=1)
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
        self.parser.add_argument("--optim", type=str, default="adam")
        self.parser.add_argument("--scheduler", type=str, default="constant_with_warmup")
        self.parser.add_argument("--weight_decay", type=float, default=0.1)
        self.parser.add_argument("--fixed_lr", action="store_true")
        self.parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training",
        )
        self.parser.add_argument(
            "--bf16",
            action="store_true",
            help="Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training",
        )
        self.parser.add_argument(
            "--ds_config",
            default=None,
            type=str,
            help="""Optional path to deepspeed config.""",
        )

    def add_eval_options(self):
        self.parser.add_argument("--write_results", action="store_true", help="save results")
        self.parser.add_argument(
            "--eval_subset",
            default=0,
            type=int,
            help="If positive, evaluation is carried out on a subset of the chosen size.",
        )
        self.parser.add_argument(
            "--eval_freq",
            type=int,
            default=500,
            help="evaluate model every <eval_freq> steps during training",
        )
        self.parser.add_argument(
            "--save_freq",
            type=int,
            default=5000,
            help="save model every <save_freq> steps during training",
        )
        self.parser.add_argument(
            "--eval_print_freq",
            type=int,
            default=1000,
            help="print intermdiate results of evaluation every <eval_print_freq> steps",
        )

    def add_reader_options(self):
        self.parser.add_argument(
            "--model_size", type=str, default="base", choices=["base", "large", "3b", "6b", "11b"]
        )
        self.parser.add_argument(
            "--text_maxlength",
            type=int,
            default=512,
            help="maximum number of tokens in text segments (question+passage)",
        )
        self.parser.add_argument(
            "--answer_maxlength",
            type=int,
            default=-1,
            help="maximum number of tokens used to train the model, no truncation if -1",
        )
        self.parser.add_argument(
            "--max_contexts",
            type=int,
            default=8,
            help="maximum number of contexts fo training and evaluation",
        )

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument(
            "--name", type=str, default="fid_tmp", help="name of the experiment"
        )
        self.parser.add_argument(
            "--model_path", type=str, default=None, help="path for retraining"
        )
        self.parser.add_argument(
            "--cache_path",
            default="./tmp-cache/",
            type=str,
            help="Path for cache.",
        )
        self.parser.add_argument(
            "--savedir_base",
            type=str,
            help="base directory to save outputs (--name's value is appended)",
        )

        # dataset parameters
        self.parser.add_argument(
            "--per_gpu_batch_size",
            default=2,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )
        self.parser.add_argument(
            "--dataset_name",
            default="dssk_training_data",
            type=str,
            help="Name of the original dataset used to build the task. Do not confuse 'name' with 'path'! If you want to use 'ServiceNow/foo', the 'name' is 'foo'!",
        )
        self.parser.add_argument(
            "--train_datasets",
            default="msmarco;nq;topiocqa;hotpotqa;squad_v2",
            type=str,
            help="Names of datasets to use for training, spaced by ';'. Only their train split will be used.",
        )
        self.parser.add_argument(
            "--val_datasets",
            default="nq;topiocqa;hotpotqa",
            type=str,
            help="Names of datasets to use for validation, spaced by ';'. Only their val split will be used.",
        )

        self.parser.add_argument(
            "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
        )

        self.parser.add_argument(
            "--seed", type=int, default=0, help="random seed for initialization"
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Debug mode",
        )

    def parse(self, explicit_arguments):
        opt, _ = self.parser.parse_known_args(explicit_arguments)
        return opt


def get_options(use_reader=False, use_optim=False, use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
