from typing import Any, Optional
from datasets import Dataset

from dssk.utils.hf_datasets import update_infodict


def get_single_context_with_trivial_strategy(d: dict[str, Any]) -> str:
    return " ".join(d["contexts_list"])


def cross_colon_format(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    """For cross-attention models using 'question: ' etc. as Ersatz for special tokens
    This is deprecated.
    """
    answer = d.get("answer", "")
    if answered_example:
        assert answer  # Both None and "" are illegal.
    else:
        answer = ""
    context = get_single_context_with_trivial_strategy(d)
    return {
        "self_input_str": f"question: {d['question']} \n answer: {answer}",
        "cross_input_str": f"context: {context}",
    }


def cross_user_assistant_format(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    """For cross-attention models with a base decoder using a <|user|> <|assistant|> template."""
    answer = d.get("answer", "")
    if answered_example:
        assert answer  # Both None and "" are illegal.
    else:
        answer = ""
    context = get_single_context_with_trivial_strategy(d)
    return {
        "self_input_str": f"<|user|>\n|<Q>|{d['question']}\n<|assistant|>\n{answer}",
        "cross_input_str": f"<|user|>\n|<C>|\n<|assistant|>\n{context}",
    }


def system_user_assistant_prompt_format(
    d: dict[str, Any], answered_example: bool
) -> dict[str, Any]:
    answer = d.get("answer", "")
    if answered_example:
        assert answer  # Both None and "" are illegal.

    # TODO: Revisit this formatting choice (only one context was tested, and not extensively).
    combined_context = "; ".join(context_text for context_text in d["contexts_list"])
    prefix = f"<|system|>\n{combined_context}\n<|end|>\n" if combined_context else ""
    input_str = f"{prefix}<|user|>\n{d['question']}\n<|end|>\n<|assistant|>"
    if answered_example:
        input_str = f"{input_str}{answer}<|end|>"
    return {"input_str": input_str}


def tulu2_prompt_format(
    d: dict[str, Any], answered_example: bool, include_context: bool = True
) -> dict[str, Any]:
    """Native format of tulu v2 models (NOT for cross-attending models!)

    Format described here https://huggingface.co/allenai/tulu-2-dpo-7b
    """
    answer = d.get("answer", "")
    if answered_example:
        assert answer  # Both None and "" are illegal.
    # Using just a space as the separator
    combined_context = " ".join(context_text for context_text in d["contexts_list"])
    prefix = f"<|system|>\n{combined_context}\n" if (combined_context and include_context) else ""
    input_str = f"{prefix}<|user|>\n{d['question']}\n<|assistant|>\n"
    if answered_example:
        input_str = f"{input_str}{answer}"
    return {"input_str": input_str}


def tulu2_prompt_format_no_context(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    """Same as tulu2_prompt_format, but without including the context in the prompt."""
    return tulu2_prompt_format(d, answered_example=answered_example, include_context=False)


def fid_format(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    if answered_example:
        raise NotImplementedError(
            "Answered examples (typically for training) are not implemented yet."
        )

    if d["contexts_list"]:
        passages = [
            f"question: {d['question']} title: {title} context: {context}"
            for context, title in zip(d["contexts_list"], d["titles_list"])
        ]
    else:
        passages = [f"question: {d['question']}"]

    return {"passages": passages}


KNOWN_QA_TASK_FORMATS = {
    "cross_colon": cross_colon_format,
    "cross_uaf": cross_user_assistant_format,
    "prompt": system_user_assistant_prompt_format,
    "prompt_tulu2": tulu2_prompt_format,
    "prompt_tulu2_no_context": tulu2_prompt_format_no_context,
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
    ),
    "cross": (
        None,
        {"self_input_str", "cross_input_str", "context"},
    ),
    "tulu2": (
        None,
        {"input_str", "context"},
    ),
    "fid": (
        None,
        {"passages", "context"},
    ),
}


class CleanupQATask:
    def __init__(self, *, post_cleanup: Optional[str] = None, **kwargs):
        self.post_cleanup = post_cleanup

    def __call__(self, qa_task: Dataset) -> Dataset:
        if self.post_cleanup:
            formatter, dropped = KNOWN_POST_CLEANUPS[self.post_cleanup]
            # Silently ignore dropping columns that don't exist
            dropped = set(qa_task.column_names) & set(dropped)
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
