# Copyright 2024 ServiceNow
# This file is based on code by the authors denoted below and has been modified from its original version.
#
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
# The following file is heavily based on the `instruct_qa` repo
# https://github.com/McGill-NLP/instruct-qa/blob/main/experiments/calculate_scores.py
###
####################################

"""
Given an inference dataset, compute scores.
The dataset is a HF dataset with the following info.

DatasetInfo(
  description='{
      'eval_task': {
       'dataset_name': 'long_nq_dedup',
       'dataset_split': 'test_answered',
       'context': 'only_gold_long',
       'answer': 'newline',
       'subset_size': None
   },
   'format': {
       'task_format': 'cross',
       'answered_example': False
   },
   'model': {
       'class_name': 'CrossAttnGPTBigCode',
       'name_or_path': 'data_rw/hf_models/starcoderbase-3b',
       'default_gen_args': {
           'max_new_tokens': 30
       }
   }
  features={
      context: text
      contexts_list: list of text
      titles_list: list of text
      useful_contexts: list of int
      question: text
      answer: text
      sample_idx: int
      dataset: text
      answer_pred: text
      error: text
      error_msg: text
  }
)
"""

import os
import json
from datasets import load_from_disk
from typing import Optional, Any, Dict, List

from xc_cache.metrics.generation.utils import load_metric
from xc_cache.data.process_answers import KNOWN_ANSWER_PROCESSING
from xc_cache.utils.hf_datasets import merge_duplicated_rows


def get_metric_scores(
    metrics: List, predictions: List[str], references: List[str], questions: List, ids: List, args
) -> Dict:
    """
    Given a list of metrics and the relevant predictions and references,
    compute the score for each metric and return in dict format.
    """
    scores = {}
    # For each metric, compute it
    for metric_name in metrics:
        try:
            print("Calculating {}".format(metric_name))
            # file_name is only used when computing LLMEval
            metric = load_metric(metric_name, file_name=metric_name, args=args)
            scores[metric_name] = metric(
                predictions=predictions, references=references, questions=questions, ids=ids
            )
            print("{} = {}".format(metric_name, scores[metric_name]))
        except Exception as error:
            print(f"***************** Error processing {metric_name}: {error}")
            continue

    return scores


def get_faith_metric_scores(
    faith_metrics: List, history: List, response: List, evidence: List, ids: List, args
) -> Dict:
    """
    Given a list of faithfulness metrics and the relevant predictions and references,
    compute the score for each metric and return in dict format.
    """
    scores = {}
    # For each metric, compute it
    for metric_name in faith_metrics:
        try:
            print("Calculating {}".format(metric_name))
            # file_name is only used when computing LLMEval
            metric = load_metric(metric_name, file_name=metric_name, args=args)
            scores[metric_name] = metric(
                history_list=history, response_list=response, evidence_list=evidence, ids=ids
            )
            print("{} = {}".format(metric_name, scores[metric_name]))
        except Exception as error:
            print(f"***************** Error processing {metric_name}: {error}")
            continue

    return scores


def get_score_filename(
    dataset_desc_dict: Dict,
    dataset: Optional[str] = None,
    best_answer: bool = False,
    topic_retrieval: bool = False,
) -> str:
    """
    Given the dataset info, generate a json filename to save the scores to.
    If a dataset is provided, include it in the filename.
    Assumes the dataset_desc_dict contains ['model']['class_name'] and ['model']['model_name']
    """
    class_name = dataset_desc_dict.get("model", {}).get("class_name", "class")
    model_name = dataset_desc_dict.get("model", {}).get("name_or_path", "")

    dataset_name = "all_test" if dataset is None else dataset

    # Try to find the model name elsewhere
    if model_name == "":
        model_name = dataset_desc_dict.get("model", {}).get("class_name", "")
    else:
        model_name = os.path.split(
            dataset_desc_dict.get("model", {}).get("name_or_path", "").strip("/")
        )[-1]

    model_name = f"{model_name}_" if model_name else ""
    class_name = f"{class_name}_" if class_name else ""
    best_ans = "best_answer_" if best_answer else ""
    topic_ret = "_topic_retrieval_" if topic_retrieval else ""
    return f"eval_scores_{class_name}{model_name}_{best_ans}{topic_ret}{dataset_name}.json"


def load_and_filter_data(file_name: str, args: Optional[Dict[str, Any]] = None):
    """
    Load the dataset from the given file_name.
    If filters are given, then filter before returning the dataset.
    Expected arguments:
        - dataset_filter_idx_file
        - best_answer
        - dataset_name
    """
    # Load the dataset
    inf_dataset = load_from_disk(file_name)

    # If a filter is specified, select the samples based on these filter indices
    # The filter_idx_file is a txt file containing a list of indices
    if args.dataset_filter_idx_file is not None:
        # Reading the list of integers from the file
        filter_idx = []
        with open(args.dataset_filter_idx_file, "r") as file:
            for line in file:
                filter_idx.append(int(line.strip()))
        inf_dataset = inf_dataset.select(filter_idx)

    # If a dataset name is given, filter the data
    if args.dataset_name is not None:
        inf_dataset = inf_dataset.filter(lambda example: example["dataset"] == args.dataset_name)
        print(f"Filtering out samples from dataset {args.dataset_name}")
        if inf_dataset.shape[0] == 0:
            raise Exception(f"No samples were found from dataset {args.dataset_name}! Aborting.")

    print(f"Loaded dataset with {inf_dataset.shape[0]} samples.")

    # If a we want to compute metrics only for samples with an answer, filter
    if args.only_answerable:
        inf_dataset = inf_dataset.filter(
            lambda example: example["answer"].lower() != "unanswerable"
        )
        print("Selecting only samples that are answerable *****************", inf_dataset.shape[0])
    # If a we want to compute metrics based on best-answer matching with GT answer
    if args.best_answer:
        inf_dataset = merge_duplicated_rows(inf_dataset, ("answer",), ("sample_idx",))
        print(
            "Grouping data samples by question/context pairs for `best answer` metric computation."
        )

    return inf_dataset


def calculate_score_for_single_file(
    file_name: str,
    metrics: Optional[Dict[str, Any]] = None,
    faith_metrics: Optional[Dict[str, Any]] = None,
    args: Optional[Dict[str, Any]] = None,
):
    assert (metrics is not None) or (faith_metrics is not None)
    assert (len(metrics) > 0) or (len(faith_metrics) > 0)

    # Load the dataset and apply the filters
    inf_dataset = load_and_filter_data(file_name, args)

    print(f"Evaluating dataset with {inf_dataset.shape[0]} samples .....")

    # Initialize the scores
    scores = faith_scores = {}

    # Setup the data lists needed for metric computations
    if inf_dataset.features.get("sample_idx", None) is None:
        ids = [i for i in range(inf_dataset.shape[0])]
    else:
        ids = inf_dataset["sample_idx"]
    questions = inf_dataset["question"]

    # If evaluating topic retrieval, pick the right column
    if args.topic_retrieval:
        answer = inf_dataset["document_category"]
        # answer_pred = inf_dataset["answer_pred_retrieved_topic"]
        answer_pred = inf_dataset["cleaned_answer_retrieved_topic"]
    else:
        # If the flag is set, then we want the GT answer for all samples to be UNANSWERABLE
        # This is the case when evaluating in the no-ICL setting
        if args.GT_UNANSWERABLE:
            answer = ["UNANSWERABLE" for x in range(inf_dataset.shape[0])]
        else:
            answer = inf_dataset["answer"]
        answer_pred = inf_dataset["answer_pred"]
        # answer_pred = inf_dataset["full_predicted_answer"]
        # answer_pred = inf_dataset["cleaned_answer_pred"]

    # Predicted answer is post-processed if the argument is passed
    if args.answer_processing:
        answer_pred = KNOWN_ANSWER_PROCESSING[args.answer_processing](answer_pred)

    # Reference answers are either single-answer, or a list of answers depending on
    # whether we want best_match metric computation or not.
    if args.best_answer:
        references = inf_dataset["answer_list"]
    elif (
        isinstance(answer, List)
        and isinstance(answer[0], Dict)
        and (answer[0].get("aliases", None) is not None)
    ):
        # If the answers are a dictionary, then assume the references are a list of answers.
        references = [ans["aliases"] for ans in answer]
    else:
        # The metric code requires a list of possible references for each prediction, so convert
        # the List[str] into a List[List[str]]

        # If the inf_dataset["answer"] is a dict, use the normalized text
        if type(inf_dataset["answer"][0]) is dict:
            references = [ref["normalized_aliases"] for ref in inf_dataset["answer"]]
        else:
            references = [[ref] for ref in inf_dataset["answer"]]

    # Compute the scores
    if (metrics is not None) and (len(metrics) > 0):
        scores = get_metric_scores(
            metrics=metrics,
            predictions=answer_pred,
            references=references,
            questions=questions,
            ids=ids,
            args=args,
        )

    if (faith_metrics is not None) and (len(faith_metrics) > 0):
        # history is only used for conversational datasets.
        history = inf_dataset["answer"]
        evidence = [".".join(x[0]) for x in inf_dataset["contexts_list"]]
        faith_scores = get_faith_metric_scores(
            faith_metrics=faith_metrics,
            history=history,
            response=answer_pred,
            evidence=evidence,
            ids=ids,
            args=args,
        )

    # Setup the json file we will save together with the scores
    # if inf_dataset.info.description is empty, catch the exception and continue
    try:
        output_json = json.loads(inf_dataset.info.description)
    except ValueError:
        output_json = {}
        pass

    # Make sure the save path exists.
    os.makedirs(args.score_dir, exist_ok=True)
    # Setup the full path of the json file to save
    save_json_path = os.path.join(
        args.score_dir,
        get_score_filename(output_json, args.dataset_name, args.best_answer, args.topic_retrieval),
    )

    # If no json with the same filename already exits,
    # create a new one. Otherwise load it.
    if os.path.isfile(save_json_path):
        # Get the existing file and update its info
        with open(save_json_path, "r") as file:
            output_json = json.load(file)

    # Get the correct score key
    # If "only_answerble", use "scores_answerable" as key
    if args.only_answerable:
        scores_key = "scores_answerable"
    elif args.GT_UNANSWERABLE:
        scores_key = "scores_gt_unanswerable"
    else:
        scores_key = "scores"
    # scores_key = "scores_answerable" if args.only_answerable else "scores"

    # Update the scores
    if scores_key not in output_json:
        output_json[scores_key] = {}
    if "faith_scores" not in output_json:
        output_json["faith_scores"] = {}

    output_json[scores_key].update(scores)
    output_json["faith_scores"].update(faith_scores)

    # If dataset is specified, save dataset name in dict
    output_json["score_dataset"] = "all_test" if args.dataset_name is None else args.dataset_name

    # save number of samples
    output_json["samples"] = inf_dataset.shape[0]

    # Save the updated data back to the same file
    with open(save_json_path, "w") as file:
        json.dump(output_json, file, indent=2)

    print(f"Scores were saved at {save_json_path}")

    return scores


def create_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--score_dir", default=None, type=str, help="Path where to dump results")

    parser.add_argument(
        "--inf_dataset_path",
        type=str,
        help="Path to HF dataset containing inference results.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Options are one of nq, hotpotqa or topiocqa. When set, metrics are computed on samples from this dataset.",
    )

    parser.add_argument(
        "--dataset_filter_idx_file",
        type=str,
        default=None,
        help="A file containing a list of indices (one on each line). They will be loaded as list, and used to select only the samples at these indices.",
    )

    parser.add_argument(
        "--best_answer",
        action="store_true",
        help="If set, then the samples will first be grouped by (question, context) pairs, and the GT answers of each group stored in 'answer_list' \
                The metrics are then computed by comparing the predicted answer of each sample to each one of the answers and taking the max.",
    )

    parser.add_argument(
        "--store_individual_scores",
        action="store_true",
        help="Save the score of individual samples.",
    )

    parser.add_argument(
        "--answer_processing",
        action="store",
        type=str,
        help="If set, use the appropriate processing method on the answers before computing the metrics.",
    )

    # Add individual metrics
    parser.add_argument(
        "--meteor",
        action="store_true",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
    )
    # parser.add_argument(
    #    "--bem",
    #    action="store_true",
    # )
    parser.add_argument(
        "--rouge",
        action="store_true",
    )
    parser.add_argument(
        "--bleu",
        action="store_true",
    )
    parser.add_argument(
        "--f1",
        action="store_true",
    )
    parser.add_argument(
        "--em",
        action="store_true",
    )
    parser.add_argument(
        "--recall",
        action="store_true",
    )
    parser.add_argument(
        "--recallem",
        action="store_true",
    )
    parser.add_argument(
        "--precision",
        action="store_true",
    )
    parser.add_argument(
        "--llm_eval",
        action="store_true",
    )

    # Faithfulness metrics
    parser.add_argument(
        "--kbertscore",
        action="store_true",
    )
    parser.add_argument(
        "--kprecision",
        action="store_true",
    )
    parser.add_argument(
        "--kprecision_plus_plus",
        action="store_true",
    )
    parser.add_argument(
        "--krecall",
        action="store_true",
    )
    parser.add_argument(
        "--krecall_plus_plus",
        action="store_true",
    )
    # parser.add_argument(
    #    "--kllm_eval",
    #    action="store_true",
    # )
    # parser.add_argument(
    #    "--kllm_eval_conv",
    #    action="store_true",
    # )
    parser.add_argument(
        "--kf1",
        action="store_true",
    )
    parser.add_argument(
        "--kf1_plus_plus",
        action="store_true",
    )
    parser.add_argument(
        "--all_faith_metrics",
        action="store_true",
    )

    # These are needed for some of the metrics that make calls to external LLMs
    parser.add_argument(
        "--api_key",
        action="store",
        type=str,
        default=None,
        help="API key if generating from OpenAI model.",
    )
    parser.add_argument(
        "--model_name",
        action="store",
        type=str,
        default="gpt-3.5-turbo",
        help="The OpenAI model to be used for LLMEval",
    )
    parser.add_argument(
        "--max_tokens",
        action="store",
        type=int,
        default=2,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        action="store",
        type=float,
        default=0.0,
        help="The temperature to use during generation. This is the randomness of the model's output. A higher temperature (e.g., 0.8) makes the output more diverse and creative, while a lower temperature (e.g., 0.2) makes the output more focused and deterministic.",
    )
    parser.add_argument(
        "--top_p",
        action="store",
        type=float,
        default=1.0,
        help="This parameter controls the diversity of the model's output by setting a threshold for the cumulative probability of the most likely tokens. It helps in avoiding very low probability tokens and encourages the model to generate more diverse responses..",
    )
    parser.add_argument(
        "--n",
        action="store",
        type=int,
        default=1,
        help="Number of completions to generate for each prompt",
    )
    parser.add_argument(
        "--stop_seq",
        action="store",
        type=str,
        default="",
        help="When to stop generation",
    )
    parser.add_argument(
        "--presence_penalty",
        action="store",
        type=float,
        default=0.0,
        help="Positive values increases model's likelihood to talk about new topics",
    )
    parser.add_argument(
        "--frequency_penalty",
        action="store",
        type=float,
        default=0.0,
        help="Positive values decreases model's likelihood to repeat same line verbatim",
    )

    # If set, then all available metrics will be computed
    parser.add_argument(
        "--all_metrics",
        action="store_true",
    )
    # If set, the metrics used in the ACL paper will be used.
    parser.add_argument(
        "--acl_metrics",
        action="store_true",
    )
    # If set, the metrics used in the NeurIPS paper will be used.
    parser.add_argument(
        "--neurips_metrics",
        action="store_true",
    )
    # If set, the metrics will be computed for the topic retrieval task
    parser.add_argument(
        "--topic_retrieval",
        action="store_true",
    )
    # If set, the metrics will be computed for all samples, but where the GT is "UNANSWERABLE".
    # This is useful to evaluate models in the NO-ICL setting
    parser.add_argument(
        "--GT_UNANSWERABLE",
        action="store_true",
    )
    # If set, the metrics will be computed for the samples that had a normal answr (NOT "UNANSWERABLE")
    parser.add_argument(
        "--only_answerable",
        action="store_true",
    )
    return parser


def main(explicit_arguments: Optional[list[str]] = None) -> None:
    parser = create_parser()
    args, _ = parser.parse_known_args(explicit_arguments)

    if not args.inf_dataset_path:
        raise Exception("You must specify an inference dataset!")

    # If no score directory specified, save in dataset dir
    if not args.score_dir:
        args.score_dir = args.inf_dataset_path

    all_faith_metrics = {
        "kbertscore": args.kbertscore,
        "kprecision": args.kprecision,
        "kprecision++": args.kprecision_plus_plus,
        "krecall": args.krecall,
        "kf1": args.kf1,
        "kf1++": args.kf1_plus_plus,
        "krecall++": args.krecall_plus_plus,
        # "kllm_eval": args.kllm_eval,
        # "kllm_eval_conv": args.kllm_eval_conv,
    }

    all_metrics = {
        "em": args.em,
        "precision": args.precision,
        "recall": args.recall,
        "f1": args.f1,
        "recallem": args.recallem,
        "rouge": args.rouge,
        "bleu": args.bleu,
        "meteor": args.meteor,
        "bertscore": args.bertscore,
        # "bem": args.bem, # We need to install TFlow if we want this
        "llm_eval": args.llm_eval,
        # "llm_eval_conv": args.llm_eval_conv,
    }

    acl_metrics = {
        "em": args.em,
        "precision": args.precision,
        "recall": args.recall,
        "f1": args.f1,
        "recallem": args.recallem,
        "rouge": args.rouge,
        "meteor": args.meteor,
        "bertscore": args.bertscore,
    }

    neurips_metrics = {
        "em": args.em,
        "precision": args.precision,
        "recall": args.recall,
        "f1": args.f1,
        "recallem": args.recallem,
        "bertscore": args.bertscore,
        "rouge": args.rouge,
        "meteor": args.meteor,
    }

    # Setup the metrics
    if args.all_metrics:
        metrics = list(all_metrics.keys())
        # LLM eval will be done on a subset, so we don't include it here to prevent accidental runs
        metrics.remove("llm_eval")
        metrics.remove("bertscore")
        # metrics.remove("llm_eval_conv")
    elif args.acl_metrics:
        metrics = list(acl_metrics.keys())
    elif args.neurips_metrics:
        metrics = list(neurips_metrics.keys())
        # For topic retrieval task, remove bertscore comuptation
        if args.topic_retrieval:
            metrics.remove("bertscore")
            metrics.remove("meteor")
            metrics.remove("rouge")
    else:
        metrics = []
        for m, m_arg in all_metrics.items():
            if m_arg:
                metrics.append(m)

    # Setup the faithfulness metrics
    if args.all_faith_metrics:
        faith_metrics = list(all_faith_metrics.keys())
        # LLM eval will be done on a subset, so we don't include it here to prevent accidental runs
        # faith_metrics.remove("kllm_eval")
        # faith_metrics.remove("kllm_eval_conv")
    else:
        faith_metrics = []
        for m, m_arg in all_faith_metrics.items():
            if m_arg:
                faith_metrics.append(m)

    # Show a message if we try to compute all metrics
    if args.all_metrics or args.all_faith_metrics:
        print(
            "**** NOTE **** LLM Eval will not be included in the computed metrics to avoid costly runs. Run it separately with --llm_eval."
        )

    # Make sure at least one metric is specified
    if (len(metrics) == 0) and (len(faith_metrics) == 0):
        raise ("You must specify at least one metric, or set the --all_metrics flag.")

    # sanity check
    # if any(item in ['llm_eval', 'kllm_eval', 'kllm_eval_conv', 'llm_eval_conv'] for item in metrics):
    if "llm_eval" in metrics:
        print("Note that this will run LLM eval, which will make api calls to OpenAI.")

    print("Metrics to be calculated:", metrics + faith_metrics)

    if os.path.exists(args.inf_dataset_path):
        print("Calculating the scores for dataset ", args.inf_dataset_path)
        calculate_score_for_single_file(
            args.inf_dataset_path,
            metrics=metrics,
            faith_metrics=faith_metrics,
            args=args,
        )
    else:
        raise FileNotFoundError(f"The file path '{args.inf_dataset_path}' does not exist.")


if __name__ == "__main__":
    main()
