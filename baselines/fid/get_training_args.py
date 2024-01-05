import os
from transformers import TrainingArguments

from baselines.fid.options import Options


def get_training_args(
    opt: Options, savedir: str, local_rank: int = 0, world_size: int = 1
) -> TrainingArguments:
    """Prepare training args given command line args.

    Args:
        opt: experiment config
        savedir: Folder where to save experiment data.
        local_rank: rank of executing node. 0 for principal.
        world_size: number of nodes

    Returns:
        TrainingArguments: Training arguments for HF trainer.
    """

    n_workers = 0 if opt.debug else max(os.cpu_count() // world_size, 1)

    training_args = TrainingArguments(
        remove_unused_columns=False,  # This needs to be False since our custom dataset passes fields that are processed in the collator.
        # optimization
        per_device_train_batch_size=opt.per_gpu_batch_size,
        per_device_eval_batch_size=opt.per_gpu_batch_size,
        max_steps=opt.total_steps,
        learning_rate=opt.lr,
        lr_scheduler_type=opt.scheduler,
        warmup_ratio=min(1, opt.warmup_steps / opt.total_steps),
        weight_decay=opt.weight_decay,
        max_grad_norm=opt.clip,
        gradient_accumulation_steps=opt.accumulation_steps,
        dataloader_num_workers=n_workers,
        fp16=opt.fp16,
        bf16=opt.bf16,
        # logging and checkpointing
        report_to="wandb",
        logging_dir=os.path.join(savedir, "logs"),
        logging_strategy="steps",
        logging_steps=opt.eval_freq,
        logging_first_step=True,
        output_dir=savedir,
        log_on_each_node=False,  # Only rank 0 will write logs.
        # save_total_limit=5,  # save at most 5 checkpoints (last 4 + best)
        save_strategy="steps",
        save_steps=opt.save_freq,
        load_best_model_at_end=True,
        # evaluation
        eval_steps=opt.eval_freq,
        evaluation_strategy="steps",
        # distributed
        ddp_find_unused_parameters=False,
        deepspeed=opt.deepspeed,
        local_rank=local_rank,
    )

    return training_args
