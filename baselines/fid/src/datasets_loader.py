# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import torch
from random import shuffle

from torch.utils.data import RandomSampler, Sampler


class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        """Sample batches from an aggregated dataset such that each batch contains only samples from a single dataset.

        Args:
            data_source (datasets.Dataset): aggregated dataset
            batch_size (int): number of sampler per a batch
        """

        # hard-coded to avoid iterating through the whole dataset
        self.dataset_names = ["msmarco", "nq", "topiocqa", "hotpotqa", "squad_v2"]

        # a sampler per dataset, Filtering is slow but it's cached
        self.samplers = [
            RandomSampler(data_source.filter(lambda example: example["dataset"] == name))
            for name in self.dataset_names
        ]
        self.batch_size = batch_size
        self.data_source = data_source

    def __iter__(self):
        """Sequentially iterates over the samplers yielding a batch of data exclusively from one dataset

        Yields:
            dict: one sample at a time
        """

        # list of counters of sampled data by sampler to keep track of when they are exhausted
        remaining = [len(sampler) for sampler in self.samplers]

        # extra security: the dataloader that calls this batchsampler should stop after sampling self.__len__() data
        while all([r >= self.batch_size for r in remaining]):
            # sequentially sample from each dataset
            for i, sampler in enumerate(self.samplers):
                # skip sampler if it doe not have enough data for a full batch
                if remaining[i] >= self.batch_size:
                    # change sampler when a full batch has been sampled
                    for _ in range(self.batch_size):
                        remaining[i] -= 1
                        yield next(sampler.__iter__())

    def __len__(self):
        """Accounts for data that is dropped by each sampler as it cannot form a full batch

        Returns:
            int: number of sampled data
        """

        return self.batch_size * sum(
            [len(sampler) // self.batch_size for sampler in self.samplers]
        )


def encode_passages(batch_text_passages, tokenizer, text_maxlength=None):
    passage_ids, passage_masks = [], []
    max_n_contexts = 0
    for text_passages in batch_text_passages:
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=text_maxlength,
            padding=True,
            return_tensors="pt",
            truncation=text_maxlength is not None,
        )
        passage_ids.append(p["input_ids"][None])
        passage_masks.append(p["attention_mask"][None])
        max_n_contexts = max(max_n_contexts, len(text_passages))

    passage_ids = torch.nested.nested_tensor(passage_ids).to_padded_tensor(0).squeeze(1)
    passage_masks = torch.nested.nested_tensor(passage_masks).to_padded_tensor(0).squeeze(1)
    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def _get_target(self, example):
        ans = example.get("answer", None)
        return ans
        # return target + " </s>" no need to add eos, already added by tokenizer

    def _format_input(self, example):
        n_contexts = len(example["contexts_list"])
        if n_contexts < 1:
            passages = [f"question: {example['question']}"]

        else:
            # shuffle to be robust to variation in gold context position
            ctx_index = list(range(n_contexts))
            shuffle(ctx_index)

            passages = [
                f"question: {example['question']} title: {example['titles_list'][i]} context: {example['contexts_list'][i]}"
                for i in ctx_index
            ]

        return passages

    def __call__(self, batch_list):
        index = torch.tensor([ex["sample_idx"] for ex in batch_list])
        target = [self._get_target(ex) for ex in batch_list]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding=True,
            return_tensors="pt",
            truncation=self.answer_maxlength > 0,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        text_passages = [self._format_input(example) for example in batch_list]
        passage_ids, passage_masks = encode_passages(
            text_passages, self.tokenizer, self.text_maxlength
        )

        return (index, target_ids, target_mask, passage_ids, passage_masks)
