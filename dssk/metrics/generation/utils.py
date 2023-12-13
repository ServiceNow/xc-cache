####################################
###
# The following file is copied from the instuct_qa repo
# https://github.com/McGill-NLP/instruct-qa/blob/main/instruct_qa/evaluation/utils.py
###
####################################

from dssk.metrics.generation.metrics import (
    Bleu,
    BERTScore,
    # BEMScore, # We will need TF
    EM,
    F1,
    LLMEval,
    # LLMEvalConv,
    Meteor,
    Recall,
    RecallEM,
    Precision,
    Rouge,
    # FaithDialCriticInverse
)

from dssk.metrics.generation.faithfulness_metrics import (
    KBERTScore,
    KF1,
    KF1PlusPlus,
    KPrecision,
    KPrecisionPlusPlus,
    KRecall,
    KRecallPlusPlus,
    # KLLMEval , KLLMEvalConv, FaithDialCritic
)


def load_metric(name, file_name=None, args=None):
    metric_mapping = {
        "meteor": Meteor,
        "rouge": Rouge,
        "f1": F1,
        "bleu": Bleu,
        "em": EM,
        "recall": Recall,
        "recallem": RecallEM,
        "precision": Precision,
        "bertscore": BERTScore,
        # "bem": BEMScore,
        "llm_eval": LLMEval,
        # "llm_eval_conv": LLMEvalConv,
        # "faithcritic": FaithDialCritic,
        "kbertscore": KBERTScore,
        "kprecision": KPrecision,
        "kprecision++": KPrecisionPlusPlus,
        "krecall": KRecall,
        "kf1": KF1,
        "kf1++": KF1PlusPlus,
        "krecall++": KRecallPlusPlus,
        # "faithcritic_inverse": FaithDialCriticInverse,
        # "kllm_eval": KLLMEval,
        # "kllm_eval_conv": KLLMEvalConv,
    }

    if name not in metric_mapping:
        raise ValueError(f"{name} is not a valid metric.")

    return metric_mapping[name](name, file_name=file_name, args=args)
