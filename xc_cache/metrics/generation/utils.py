# Copyright 2023 Vaibhav Adlakha and Xing Han Lu
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####################################
###
# The following file is copied from the instuct_qa repo
# https://github.com/McGill-NLP/instruct-qa/blob/main/instruct_qa/evaluation/utils.py
###
####################################

from xc_cache.metrics.generation.metrics import (
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

from xc_cache.metrics.generation.faithfulness_metrics import (
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
