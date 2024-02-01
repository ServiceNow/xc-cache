# Code adapted from https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/utils.py

import torch

from transformers import EvalPrediction, T5Tokenizer
from typing import Callable, Dict, Iterable, List, Tuple

from dssk.metrics.generation.metrics import F1, EM


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def build_compute_metrics_fn(tokenizer: T5Tokenizer) -> Callable[[EvalPrediction], Dict]:
    f1_metric = F1("F1")
    em_metric = EM("EM")

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_ids = pred.predictions[0].argmax(axis=2)
        label_ids = pred.label_ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def compute_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        # these metrics expect a list of reference labels for each example
        scores = f1_metric(pred_str, [[labels] for labels in label_str])
        scores |= em_metric(pred_str, [[labels] for labels in label_str])
        return scores

    return compute_metrics


class GenEvaluator:
    def __init__(
        self,
        tokenizer,
    ):
        self.f1_evaluator = F1("F1")
        self.em_evaluator = EM("EM")
        self.tokenizer = tokenizer

    def __call__(self, model, data_loader) -> float:
        """Compute generation metrics."""

        predictions = []
        targets = []

        model.eval()
        with torch.no_grad():
            for inputs in data_loader:
                labels_ids = inputs.pop("labels")

                labels_ids = labels_ids.masked_fill(
                    labels_ids == -100, self.tokenizer.pad_token_id
                )
                targets += [
                    [target]
                    for target in self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
                ]

                output = model.generate(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    max_length=30,
                )

                predictions += self.tokenizer.batch_decode(output, skip_special_tokens=True)

        metrics = {
            "eval/gen_f1": self.f1_evaluator(predictions, targets)["f1"],
            "eval/gen_em": self.em_evaluator(predictions, targets)["em"],
        }

        return metrics
