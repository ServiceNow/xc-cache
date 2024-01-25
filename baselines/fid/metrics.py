# Code adapted from https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/utils.py

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
