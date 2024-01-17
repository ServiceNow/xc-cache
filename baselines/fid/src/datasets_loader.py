# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import torch
import logging

from torch.utils.data import RandomSampler, Sampler

from dssk.data.utils.pre_processors import PosContextPreProcessor

logger = logging.getLogger(__name__)


# new class
class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size, dataset_names):
        """Sample batches from an aggregated dataset such that each batch contains only samples from a single dataset.

        Args:
            data_source (datasets.Dataset): aggregated dataset
            batch_size (int): number of sampler per a batch
        """

        # hard-coded to avoid iterating through the whole dataset
        self.dataset_names = dataset_names.split(";")

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
    max_n_tokens = 0
    for text_passages in batch_text_passages:
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=text_maxlength,
            padding=True,
            return_tensors="pt",
            truncation=text_maxlength is not None,
        )
        passage_ids.append(p["input_ids"])
        passage_masks.append(p["attention_mask"])
        # keep track of maximal number of tokens and of contexts
        max_n_contexts = max(max_n_contexts, len(text_passages))
        max_n_tokens = max(max_n_tokens, p["attention_mask"].shape[1])

    # number of contexts may vary within a batch. Need to pad that dimension
    passage_ids = torch.nested.nested_tensor(passage_ids).to_padded_tensor(0)
    passage_masks = torch.nested.nested_tensor(passage_masks).to_padded_tensor(0)

    # log statistics
    logger.debug(
        f"batch's maximal number of tokens: {max_n_tokens} and of contexts {max_n_contexts}"
    )

    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, max_contexts=10):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.max_contexts = max_contexts

    def _format_input(self, example):
        n_contexts = len(example["contexts_list"])

        if n_contexts < 1:  # no contexts provided
            passages = [f"question: {example['question']}"]

        elif n_contexts == 1:  # only one gold context provided
            passages = [
                f"question: {example['question']} title: {example['titles_list'][0]} context: {example['contexts_list'][0]}"
            ]

        else:
            # select all gold and some distractors (for a total of up to max_contexts), and shuffle them to be robust to order variations
            ctx_index = PosContextPreProcessor.truncate_true_list(
                list(range(n_contexts)), example["useful_contexts"], max_items=self.max_contexts
            )

            passages = [
                f"question: {example['question']} title: {example['titles_list'][i]} context: {example['contexts_list'][i]}"
                for i in ctx_index
            ]

        return passages

    def __call__(self, batch_list):
        target = [example.get("answer", None) for example in batch_list]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding=True,
            return_tensors="pt",
            truncation=self.answer_maxlength > 0,
        )
        target_ids = target["input_ids"]
        target_ids = target_ids.masked_fill(
            ~target["attention_mask"].bool(), -100
        )  # padding tokens are ignored when computing the loss

        text_passages = [self._format_input(example) for example in batch_list]
        passage_ids, passage_masks = encode_passages(
            text_passages, self.tokenizer, self.text_maxlength
        )

        return {
            "labels": target_ids,
            "input_ids": passage_ids,
            "attention_mask": passage_masks,
        }
