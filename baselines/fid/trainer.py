import numpy as np
import torch
import wandb

from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import DataCollator, Trainer, TrainerCallback, TrainingArguments

from baselines.fid.src.datasets_loader import BatchSampler
from baselines.fid.options import Options
from baselines.fid.src.t5_wrapper import FiDT5

from typing import Optional, Dict


class FiDTrainer(Trainer):
    """Custom trainer class for training FiD."""

    def __init__(
        self,
        model: FiDT5,
        data_collator: DataCollator,
        args: TrainingArguments,
        opt: Options,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        callbacks: Optional[TrainerCallback] = None,
    ):
        super(FiDTrainer, self).__init__(
            model=model,
            data_collator=data_collator,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        self.can_return_loss = True  # key override to log loss
        self.eval_subset = opt.eval_subset
        self.train_datasets = opt.train_datasets
        self.val_datasets = opt.val_datasets

    def log(self, logs: Dict[str, float]):
        """
        Log `logs` on the various objects watching training.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """

        # default logging
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataset = dataset.filter(
            lambda example: example["dataset"] in self.val_datasets.split(";")
        )

        if self.eval_subset > 0:
            # evaluate on subset of random samples
            subset = Subset(
                dataset, np.random.choice(len(dataset), self.eval_subset, replace=False).tolist()
            )
        else:
            subset = dataset

        if torch.distributed.is_initialized():
            sampler = DistributedSampler(subset, shuffle=False)
        else:
            sampler = SequentialSampler(subset)

        dataloader = DataLoader(
            subset,
            sampler=sampler,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            drop_last=False,
        )

        return dataloader

    def get_train_dataloader(self):
        train_sampler = BatchSampler(
            self.train_dataset, self.args.per_device_train_batch_size, self.train_datasets
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            drop_last=False,
        )

        return train_dataloader


def get_trainer(
    model: FiDT5,
    data_collator: DataCollator,
    opt: Options,
    training_args: TrainingArguments,
    training_data: Dataset,
    validation_data: Dataset,
) -> Trainer:
    """Intanstiates Trainer object.

    Args:
        model (FiDT5): Model to be trained.
        data_collator (DataCollator): Data collator.
        opt (Options): Command line arguments.
        training_args (TrainingArguments): trainer's argument.
        training_data (Dataset): Training dataset.
        validation_data (Dataset): Validation dataset.

    Returns:
        Trainer: Configured trainer.
    """

    # prepare experiment config to log in wandb
    config = {}
    for k, value in vars(opt).items():
        if isinstance(value, (int, str, bool, float)):
            config[k] = value

    # init wandb logger
    wandb.init(
        name=None,
        project=opt.name,
        mode="disabled" if opt.debug or opt.local_rank > 0 else None,  # no logging while debugging
        config=config,
    )

    # instantiate FiD-T5 trainer
    trainer = FiDTrainer(
        model,
        data_collator,
        args=training_args,
        opt=opt,
        train_dataset=training_data,
        eval_dataset=validation_data,
    )

    return trainer
