from typing import Any, Optional
from datasets import Dataset

from dssk.utils.hf_datasets import update_infodict


def get_sample_info(d: dict[str, Any], answered_example: bool) -> tuple[str, str, str]:
    question_text = d["question_text"]
    context_texts = d["context_texts"]
    context_headers = d["contexts_headers"]
    answer_text = d.get("answer_text", None)
    if answered_example:
        assert answer_text  # Both None and "" are illegal.
    else:
        answer_text = ""
    return question_text, context_texts, context_headers, answer_text


def cross_colon_format(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    """For cross-attention models using 'question: ' etc. as Ersatz for special tokens"""
    question_text, context_texts, _, answer_text = get_sample_info(d, answered_example)

    # NOTE: The space before and after the \n are "weird", but this is how the model is trained
    #       as of October. In any case, we may change this for "proper" special tokens.
    # TODO: Revisit.
    self_input_text = f"question: {question_text} \n answer: {answer_text}"
    cross_input_texts = [[f"context: {context_text}"] for context_text in context_texts]

    return {"self_input_text": self_input_text, "cross_input_texts": cross_input_texts}


def cross_user_assistant_format(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    """For cross-attention models with a base decoder using a <|user|> <|assistant|> template."""
    question_text, context_texts, answer_text = get_sample_info(d, answered_example)

    self_input_text = f"<|user|>\n|<Q>|{question_text}\n<|assistant|>\n{answer_text}"
    cross_input_texts = [
        [f"<|user|>\n|<C>|\n<|assistant|>\n{context_text}"] for context_text in context_texts
    ]
    return {"self_input_text": self_input_text, "cross_input_texts": cross_input_texts}


def system_user_assistant_prompt_format(
    d: dict[str, Any], answered_example: bool
) -> dict[str, Any]:
    question_text, context_texts, _, answer_text = get_sample_info(d, answered_example)

    # TODO: Revisit this formatting choice (only one context was tested, and not extensively).
    combined_context = "; ".join(context_text for context_text in context_texts)
    prefix = f"<|system|>\n{combined_context}\n<|end|>\n" if combined_context else ""
    self_input_text = f"{prefix}<|user|>\n{question_text}\n<|end|>\n<|assistant|>"
    if answered_example:
        self_input_text = f"{self_input_text}{answer_text}<|end|>"
    return {"self_input_text": self_input_text, "cross_input_texts": []}


def fid_format(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    question_text, context_texts, context_headers, _ = get_sample_info(d, answered_example)

    if len(context_texts) < 1:
        passages = [f"question: {question_text}"]

    passages = [
        f"question: {question_text} title: {title[0]} context: {text}"
        for text, title in zip(context_texts, context_headers)
    ]

    return {"self_input_text": passages, "cross_input_texts": []}


KNOWN_QA_TASK_FORMATS = {
    "cross": cross_colon_format,
    "prompt": system_user_assistant_prompt_format,
    "fid": fid_format,
}


def format_qa_task(
    qa_task: Dataset,
    *,
    task_format: Optional[str] = None,
    answered_example: bool = False,
    **kwargs,
) -> Dataset:
    """Format a question answering task with a specific model in mind

    At this stage, we know which model is going to perform the task, and how it want's its inputs prepared.

    When `answered_example` is `True`, the answer is included as part of self_input_text.
    This is typically used in training, or to generate solved examples for few-shot inference.

    The new approach is to just "hope" that the formatter has the right features/columns to do what
    it has to do, and that it returns the right features/columns for the model to do the same. The
    old contract is provided below for reference.

    *** OLD CONTRACT FOLLOWS ***

    The following fields are *added*:

    - self_input_text: str
        Main input to the decoder, to be auto-regressively extended to produce the output.
        We purposefully avoid the word "prompt", for it could be confused with something that comes before the question as part of self_input_text.

    - cross_input_texts: Sequence[Sequence[str]]
        Other inputs of the decoder (perhaps needing encoding) to be cross-attended to (or such).
        - In absence of any such inputs, we have `cross_input_texts == []`.
        - If there is a single input, it is stored in `cross_input_text[0][0]`.
        - The outer Sequence is semantically understood as different documents.
        - The inner Sequence is semantically understood as different "chunks".

    For a decoder-only model, everything comes in `self_input_text`, and `cross_input_texts` should be an empty list.
    For an encoder-decoder model, `cross_input_texts` typically contains one or many contexts that the decoder may cross-attend to.

    The fields `question_text`, `context_texts` and `contexts_headers` are consumed: they must be
    in the input, and they are not in the output.
    If `answer_text` is present in the input, it must remain present in the output. If `answered_example` is `True`, then `answer_text` is mandatory in the input.
    """
    if task_format:
        qa_task = qa_task.map(
            KNOWN_QA_TASK_FORMATS[task_format], fn_kwargs={"answered_example": answered_example}
        )

    update_infodict(
        qa_task,
        {
            "format": {
                "task_format": task_format,
                "answered_example": answered_example,
            }
        },
    )
    return qa_task


def map_september_format_to_december_format(d: dict[str, Any]) -> dict[str, Any]:
    assert len(d["context_texts"]) == 1
    return {
        "sample_idx": f"{d['question_id']}-{d['annotation_id']}",
        "titles_list": [d["document_title"]],
        "question": d["question_text"],
        "answer": d["answer_text"],
        "answer_pred": d["output_text"],
        "contexts_list": d["context_texts"][0],
        "useful_contexts": [1],
        "dataset": "UNKNOWN",
    }


KNOWN_POST_CLEANUPS = {
    "sept2dec": (
        map_september_format_to_december_format,
        {
            "output_text",
            "question_index",
            "question_text",
            "annotation_index_list",
            "long_answer_clean",
            "cross_input_texts",
            "yes_no_answer",
            "answer_text",
            "annotation_index",
            "long_nq_tier",
            "document_title",
            "short_answers_text",
            "question_id",
            "contexts_headers",
            "context_texts",
            "annotation_id_list",
            "annotation_id",
            "self_input_text",
            "headers_before_long_answer",
        },
    )
}


class CleanupQATask:
    def __init__(self, *, post_cleanup: Optional[str] = None, **kwargs):
        self.post_cleanup = post_cleanup

    def __call__(self, qa_task: Dataset) -> Dataset:
        if self.post_cleanup:
            formatter, dropped = KNOWN_POST_CLEANUPS[self.post_cleanup]
            qa_task = qa_task.map(formatter, remove_columns=dropped)

        update_infodict(
            qa_task,
            {
                "cleanup": {
                    "post_cleanup": self.post_cleanup,
                }
            },
        )
        return qa_task
