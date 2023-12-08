# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.save_load_model import save
from src.metrics import average_main, ems, weighted_average
from dssk.utils.scripting import set_random_seed

logger = logging.getLogger()


def train(
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
):
    # TODO: replace with wandb logging
    # if opt.is_main:
    #     try:
    #         tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
    #     except:
    #         tb_logger = None
    #         logger.warning('Tensorboard is not available.')

    set_random_seed(
        opt.global_rank + opt.seed
    )  # different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator,
    )

    _, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        save(
                            model,
                            optimizer,
                            scheduler,
                            step,
                            best_dev_em,
                            opt,
                            checkpoint_path,
                            "best_dev",
                        )
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)

                    # TODO: log in wandb
                    # if tb_logger is not None:
                    #     tb_logger.add_scalar("Evaluation", dev_em, step)
                    #     tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.0

            if opt.is_main and step % opt.save_freq == 0:
                save(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_dev_em,
                    opt,
                    checkpoint_path,
                    f"step-{step}",
                )
            if step > opt.total_steps:
                break


def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator,
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(), attention_mask=context_mask.cuda(), max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])["answers"]
                score = ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch
