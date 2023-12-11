# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import torch
from random import shuffle


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        n_context=None,
        question_prefix="question:",
        title_prefix="title:",
        passage_prefix="context:",
    ):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example["question"]
        target = self.get_target(example)

        if "ctxs" in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example["ctxs"][: self.n_context]
            passages = [f.format(c["title"], c["text"]) for c in contexts]
            scores = [float(c["score"]) for c in contexts]
            scores = torch.tensor(scores)

            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        return {
            "index": index,
            "question": question,
            "answer": target,
            "passages": passages,
            "scores": scores,
        }

    def sort_data(self):
        if self.n_context is None or "score" not in self.data[0]["ctxs"][0]:
            return
        for ex in self.data:
            ex["ctxs"].sort(key=lambda x: float(x["score"]), reverse=True)

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
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
        if "answer" in example:
            return example["answer"]
            # return target + " </s>" no need to add eos, already added by tokenizer

        else:
            return None

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
        passage_ids, passage_masks = encode_passages(text_passages, self.tokenizer)

        return (index, target_ids, target_mask, passage_ids, passage_masks)
