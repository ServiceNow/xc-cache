# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import torch
import logging

from datasets import Dataset
from torch.utils.data import RandomSampler, Sampler

from dssk.data.format_qa_task import fid_format

logger = logging.getLogger(__name__)


# new class
class BatchSampler(Sampler):
    def __init__(self, data_source: Dataset, batch_size: int, dataset_names: list[str]):
        """Sample batches from an aggregated dataset such that each batch contains only samples from a single dataset.

        Args:
            data_source (datasets.Dataset): aggregated dataset
            batch_size (int): number of sampler per a batch
        """

        # hard-coded to avoid iterating through the whole dataset
        self.dataset_names = tuple(dataset_names)

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
    passage_ids = torch.nested.nested_tensor(passage_ids).to_padded_tensor(tokenizer.pad_token_id)
    passage_masks = torch.nested.nested_tensor(passage_masks).to_padded_tensor(
        tokenizer.pad_token_id
    )

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

    def __call__(self, batch_list):
        formatted_batch = [
            fid_format(
                d,
                answered_example=True,
                include_title=True,
                include_context=True,
                max_contexts_training=self.max_contexts,
            )
            for d in batch_list
        ]

        target = self.tokenizer.batch_encode_plus(
            [f["target"] for f in formatted_batch],
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding=True,
            return_tensors="pt",
            truncation=self.answer_maxlength > 0,
        )
        target_ids = target["input_ids"]
        target_ids = target_ids.masked_fill(
            ~target["attention_mask"].bool(), -100
        )  # padding tokens are ignored when computing the loss

        passage_ids, passage_masks = encode_passages(
            [f["passages"] for f in formatted_batch], self.tokenizer, self.text_maxlength
        )

        return {
            "labels": target_ids,
            "input_ids": passage_ids,
            "attention_mask": passage_masks,
        }
