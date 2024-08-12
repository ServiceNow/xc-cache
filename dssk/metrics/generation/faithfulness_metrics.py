####################################
###
# The following file is copied from the instuct_qa repo
# https://github.com/McGill-NLP/instruct-qa/blob/main/instruct_qa/evaluation/faithfulness_metrics.py
###
####################################

from collections import Counter
import json
import os
import time
import evaluate
import openai
from openai import (
    RateLimitError,
    APIConnectionError,
    APIStatusError,
    APIError,
)

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from dssk.metrics import Metric
from dssk.data.utils.templates import HistoryTemplate, PromptTemplate


class FaithDialCritic(Metric):
    """
    FaithDialCritic is a metric that measures the faithfulness of a response to a given evidence.
    0 - faithfull
    1 - unfaithfull
    lower score is better
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/roberta-large-faithcritic", return_tensors="pt"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "McGill-NLP/roberta-large-faithcritic",
        ).cuda()

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        """
        history_list: list of list of strings (won't be used)
        response_list: list of strings
        evidence_list: list of list passages from collection - text, title, sub_title
        """

        scores = []
        for i in range(len(evidence_list)):
            evidence = evidence_list[i]
            response = response_list[i]
            evidence_string = " ".join([e for e in evidence])
            input = self.tokenizer(evidence_string, response, return_tensors="pt", truncation=True)
            input = {key: val.cuda() for key, val in input.items()}
            output_logits = self.model(**input).logits
            score = torch.softmax(output_logits, dim=1)[:, 1].item()

            scores.append(score)

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"faithcritic": score} for score in scores])

        return {"faithcritic": np.mean(scores)}


class KBERTScore(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._metric = evaluate.load("bertscore")

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [" ".join([e for e in evidence]) for evidence in evidence_list]
        scores = self._metric.compute(
            predictions=response_list, references=evidence_strings, lang="en"
        )
        if self.store_individual_scores:
            individual_scores = []
            for i in range(len(response_list)):
                individual_scores.append(
                    {
                        "precision": scores["precision"][i],
                        "recall": scores["recall"][i],
                        "f1": scores["f1"][i],
                    }
                )
            self.save_individual_scores(ids, individual_scores)
        return {
            "precision": np.mean(scores["precision"]),
            "recall": np.mean(scores["recall"]),
            "f1": np.mean(scores["f1"]),
        }


class KPrecision(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [[" ".join([e for e in evidence])] for evidence in evidence_list]
        scores = [
            self._precision(prediction, reference)
            for prediction, reference in zip(response_list, evidence_strings)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"kprecision": score} for score in scores])
        return {"kprecision": np.mean(scores)}

    def _precision(self, prediction, references):
        precision_scores = [
            self._precision_score(prediction, reference) for reference in references
        ]
        return max(precision_scores)

    def _precision_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(prediction_tokens) == 0:
            # if prediction is empty, precision is 0
            return 0

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)

        return precision


class KPrecisionPlusPlus(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [[" ".join([e for e in evidence])] for evidence in evidence_list]
        history_strings = [" ".join([e for e in history]) for history in history_list]
        scores = [
            self._precision_plusplus(prediction, reference, query)
            for prediction, reference, query in zip(
                response_list, evidence_strings, history_strings
            )
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"kprecisionplusplus": score} for score in scores])
        return {"kprecisionplusplus": np.mean(scores)}

    def _precision_plusplus(self, prediction, references, query):
        precision_scores = [
            self._precision_plusplus_score(prediction, reference, query)
            for reference in references
        ]
        return max(precision_scores)

    def _precision_plusplus_score(self, prediction, reference, query):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)
        query_tokens = self._get_tokens(query)

        prediction_tokens = [token for token in prediction_tokens if token not in query_tokens]

        if len(prediction_tokens) == 0:
            return 1.0

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(prediction_tokens) == 0:
            # if prediction is empty, precision is 0
            return 0

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)

        return precision


class KRecall(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [[" ".join([e for e in evidence])] for evidence in evidence_list]
        scores = [
            self._recall(prediction, reference)
            for prediction, reference in zip(response_list, evidence_strings)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"krecall": score} for score in scores])
        return {"krecall": np.mean(scores)}

    def _recall(self, prediction, references):
        recall_scores = [self._recall_score(prediction, reference) for reference in references]
        return max(recall_scores)

    def _recall_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0:
            # if prediction is empty, recall is one
            return 1

        if num_common == 0:
            return 0

        recall = 1.0 * num_common / len(reference_tokens)

        return recall


class KRecallPlusPlus(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [[" ".join([e for e in evidence])] for evidence in evidence_list]
        history_strings = [" ".join([e for e in history]) for history in history_list]
        scores = [
            self._recall_plusplus(prediction, reference, query)
            for prediction, reference, query in zip(
                response_list, evidence_strings, history_strings
            )
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"krecallplusplus": score} for score in scores])
        return {"krecallplusplus": np.mean(scores)}

    def _recall_plusplus(self, prediction, references, query):
        recall_scores = [
            self._recall_plusplus_score(prediction, reference, query) for reference in references
        ]
        return max(recall_scores)

    def _recall_plusplus_score(self, prediction, reference, query):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)
        query_tokens = self._get_tokens(query)

        prediction_tokens = [token for token in prediction_tokens if token not in query_tokens]

        if len(prediction_tokens) == 0:
            return 1.0

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0:
            # if prediction is empty, recall is one
            return 1

        if num_common == 0:
            return 0

        recall = 1.0 * num_common / len(reference_tokens)

        return recall


class KF1(Metric):
    """Computes average F1 score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [[" ".join([e for e in evidence])] for evidence in evidence_list]
        scores = [
            self._f1(prediction, reference)
            for prediction, reference in zip(response_list, evidence_strings)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"kf1": score} for score in scores])
        return {"kf1": np.mean(scores)}

    def _f1(self, prediction, references):
        """Computes F1 score between a prediction and a list of references.
        Take the max F1 score if there are multiple references.
        """

        f1_scores = [self._f1_score(prediction, reference) for reference in references]
        return max(f1_scores)

    def _f1_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            return int(reference_tokens == prediction_tokens)

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1


class KF1PlusPlus(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [[" ".join([e for e in evidence])] for evidence in evidence_list]
        history_strings = [" ".join([e for e in history]) for history in history_list]
        scores = [
            self._f1_plusplus(prediction, reference, query)
            for prediction, reference, query in zip(
                response_list, evidence_strings, history_strings
            )
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"kf1plusplus": score} for score in scores])
        return {"kf1plusplus": np.mean(scores)}

    def _f1_plusplus(self, prediction, references, query):
        """Computes F1 score between a prediction and a list of references.
        Take the max F1 score if there are multiple references.
        """

        f1_scores = [
            self._f1_plusplus_score(prediction, reference, query) for reference in references
        ]
        return max(f1_scores)

    def _f1_plusplus_score(self, prediction, reference, query):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)
        query_tokens = self._get_tokens(query)

        prediction_tokens = [token for token in prediction_tokens if token not in query_tokens]

        if len(prediction_tokens) == 0:
            return 1.0

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            return int(reference_tokens == prediction_tokens)

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1


class KLLMEval(Metric):
    """
    Computes score using LLMs.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        args = kwargs.get("args", None)
        self.api_key = args.api_key
        self.model = args.model_name
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.n = args.n
        self.stop = args.stop_seq
        self.presence_penalty = args.presence_penalty
        self.frequency_penalty = args.frequency_penalty
        self.wait = 10
        instruction = 'You are given a question, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction in present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence.'
        prompt = (
            instruction
            + "\n\nQuestion: {question}\n\nPrediction: {prediction}\n\nEvidence: {evidence}\n\nCompareGPT response:"
        )
        self.prompt_template = PromptTemplate(
            variables=["question", "prediction", "evidence"],
            template=prompt,
        )
        self.system_prompt = "You are CompareGPT, a machine to verify the groudedness of predictions. Answer with only yes/no."
        openai.api_key = self.api_key
        self.individual_out_dir = self.individual_out_dir + f"/{self.model}"

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        assert (
            self.store_individual_scores
        ), "LLM requires individual scores to be stored, to avoid unnecessary API calls"
        individual_scores = []
        if os.path.exists(os.path.join(self.individual_out_dir, self.file_name)):
            with open(os.path.join(self.individual_out_dir, self.file_name)) as f:
                individual_scores = f.readlines()
        num_already_calculated = len(individual_scores)

        history_list = history_list[num_already_calculated:]
        response_list = response_list[num_already_calculated:]
        evidence_list = evidence_list[num_already_calculated:]
        ids = ids[num_already_calculated:]

        if len(history_list) == 0:
            print("All scores already calculated")

        for i in range(len(evidence_list)):
            # individual score handles differently to avoid repeated calls to the API
            self._llm_score(history_list[i], response_list[i], evidence_list[i], ids[i])

        with open(os.path.join(self.individual_out_dir, self.file_name)) as f:
            individual_scores = f.readlines()
        individual_scores = [json.loads(score) for score in individual_scores]

        scores = [score[self.name][self.name] for score in individual_scores]
        result = {"llm_eval": np.mean(scores)}
        return result

    def _llm_score(self, history, response, evidence, id_):
        assert len(history) == 1
        question = history[0]
        evidence_string = " ".join([e for e in evidence])
        prompt = self.prompt_template.format(
            {"question": question, "prediction": response, "evidence": evidence_string}
        )
        response = None
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
        # Except multiple errors in one except block
        except (
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APIError,
        ) as e:
            print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self._llm_score(history, response, evidence, id_)

        response = response["choices"][0]["message"]["content"].strip().strip(".").strip(",")

        if response.lower() not in [
            "yes",
            "no",
        ]:
            print(
                f"Response {response} not in ['yes', 'no']\nSystem prompt: {self.system_prompt}\nPrompt: {prompt}"
            )
        if "yes" in response.lower():
            score = 1.0
        else:
            score = 0.0

        os.makedirs(self.individual_out_dir, exist_ok=True)
        with open(os.path.join(self.individual_out_dir, self.file_name), "a") as f:
            f.write(
                json.dumps(
                    {
                        "id_": id_,
                        "question": question,
                        "evidence": evidence_string,
                        "prediction": response,
                        self.name: {"kllm_eval": score},
                    }
                )
                + "\n"
            )


class KLLMEvalConv(KLLMEval):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        instruction = 'You are given a conversation, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction in present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence.'
        prompt = (
            instruction
            + "\n\n{conversation_history}\n\nPrediction: {prediction}\n\nEvidence: {evidence}\n\nCompareGPT response:"
        )
        self.prompt_template = PromptTemplate(
            variables=["conversation_history", "prediction", "evidence"],
            template=prompt,
        )
        self.history_template = HistoryTemplate()

    def _llm_score(self, history, response, evidence, id_):
        utterances = []
        for i, utterance in enumerate(history):
            if i % 2 == 0:
                utterances.append({"speaker": "Human", "utterance": utterance})
            else:
                utterances.append({"speaker": "Assistant", "utterance": utterance})
        serialized_conv_history = self.history_template.serialize_history(utterances)

        evidence_string = " ".join([e for e in evidence])
        prompt = self.prompt_template.format(
            {
                "conversation_history": serialized_conv_history,
                "prediction": response,
                "evidence": evidence_string,
            }
        )
        response = None
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
        # Except multiple errors in one except block
        except (
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APIError,
        ) as e:
            print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self._llm_score(history, response, evidence, id_)

        response = response["choices"][0]["message"]["content"].strip().strip(".").strip(",")

        if response.lower() not in [
            "yes",
            "no",
        ]:
            print(
                f"Response {response} not in ['yes', 'no']\nSystem prompt: {self.system_prompt}\nPrompt: {prompt}"
            )
        if "yes" in response.lower():
            score = 1.0
        else:
            score = 0.0

        os.makedirs(self.individual_out_dir, exist_ok=True)
        with open(os.path.join(self.individual_out_dir, self.file_name), "a") as f:
            f.write(
                json.dumps(
                    {
                        "id_": id_,
                        "question": serialized_conv_history,
                        "evidence": evidence_string,
                        "prediction": response,
                        self.name: {"kllm_eval": score},
                    }
                )
                + "\n"
            )
