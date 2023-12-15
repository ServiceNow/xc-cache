# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import numpy as np
import torch

from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from baselines.fid.src.datasets_loader import BatchSampler
from baselines.fid.src.metrics import average_main, ems, weighted_average
from baselines.fid.src.save_load_model import save

from dssk.utils.scripting import set_random_seed


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
    logger,
    wandb_run,
):
    set_random_seed(
        opt.local_rank + opt.seed
    )  # different seed for different sampling depending on global_rank

    train_sampler = BatchSampler(train_dataset, opt.per_gpu_batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=0 if opt.debug else opt.n_workers,
        collate_fn=collator,
        drop_last=False,
    )

    _, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1

        pbar = tqdm(train_dataloader)
        for batch in pbar:
            step += 1
            (_, labels, _, context_ids, context_mask) = batch

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

            wandb_run.log({"train/loss": train_loss.item()})

            if step % opt.eval_freq == 0:
                dev_em, val_loss = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)

                    wandb_run.log(
                        {
                            "lr": scheduler.get_last_lr()[0],
                            "val/EM": 100 * dev_em,
                            "val/loss": val_loss.item(),
                        }
                    )

                    pbar.set_description(f"Loss: {curr_loss / (opt.eval_freq)}, val EM: {dev_em}")
                    curr_loss = 0.0

            if opt.is_main and (step % opt.save_freq == 0 or step == opt.total_steps):
                if dev_em > best_dev_em:
                    best_dev_em = dev_em
                    save_name = "best_dev"
                else:
                    save_name = f"step-{step}"

                save(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_dev_em,
                    opt,
                    checkpoint_path,
                    save_name,
                )
            if step > opt.total_steps:
                break


def evaluate(model, dataset, tokenizer, collator, opt):
    if opt.eval_subset > 0:
        # evaluate on subset of random samples
        subset = Subset(
            dataset, np.random.choice(len(dataset), opt.eval_subset, replace=False).tolist()
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
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=0 if opt.debug else opt.n_workers,
        collate_fn=collator,
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            (idx, labels, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(), attention_mask=context_mask.cuda(), max_length=50
            )

            val_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
            )[0]
            total_loss += average_main(val_loss, opt)

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset[idx[k].item()]["answer"]
                score = ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch, total_loss / len(dataloader)
