# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import torch
from random import shuffle


def encode_passages(batch_text_passages, tokenizer, text_maxlength=None):
    passage_ids, passage_masks = [], []
    for text_passages in batch_text_passages:
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=text_maxlength,
            padding="max_length",
            return_tensors="pt",
            truncation=text_maxlength is not None,
        )
        passage_ids.append(p["input_ids"][None])
        passage_masks.append(p["attention_mask"][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
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
            padding="max_length",
            return_tensors="pt",
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        text_passages = [self._format_input(example) for example in batch_list]
        passage_ids, passage_masks = encode_passages(
            text_passages, self.tokenizer, self.text_maxlength
        )

        return (index, target_ids, target_mask, passage_ids, passage_masks)
