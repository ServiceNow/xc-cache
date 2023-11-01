from typing import Any
from datasets import Dataset

from dssk.utils.no_cache import no_cache


def get_sample_info(d: dict[str, Any], answered_example: bool) -> tuple[str, str, str]:
    question_text = d["question_text"]
    context_texts = d["context_texts"]
    answer_text = d.get("answer_text", None)
    if answered_example:
        assert answer_text  # Both None and "" are illegal.
    else:
        answer_text = ""
    return question_text, context_texts, answer_text


def cross_colon_format(d: dict[str, Any], answered_example: bool) -> dict[str, Any]:
    """For cross-attention models using 'question: ' etc. as Ersatz for special tokens"""
    question_text, context_texts, answer_text = get_sample_info(d, answered_example)

    # NOTE: The space before and after the \n are "weird", but this is how the model is trained
    #       as of October. In any case, we may change this for "proper" special tokens.
    # TODO: Revisit.
    self_input_text = f"question: {question_text} \n answer: {answer_text}"
    cross_input_texts = [[f"context: {context_text}"] for context_text in context_texts]

    return {"self_input_text": self_input_text, "cross_input_texts": cross_input_texts}


def system_user_assistant_prompt_format(
    d: dict[str, Any], answered_example: bool
) -> dict[str, Any]:
    question_text, context_texts, answer_text = get_sample_info(d, answered_example)

    # TODO: Revisit this formatting choice (only one context was tested, and not extensively).
    combined_context = "; ".join(context_text for context_text in context_texts)
    prefix = f"<|system|>\n{combined_context}\n<|end|>\n" if combined_context else ""
    self_input_text = f"{prefix}<|user|>\n{question_text}\n<|end|>\n<|assistant|>"
    if answered_example:
        self_input_text = f"{self_input_text}{answer_text}<|end|>"
    return {"self_input_text": self_input_text, "cross_input_texts": []}


KNOWN_QA_TASK_FORMATS = {
    "cross": cross_colon_format,
    "prompt": system_user_assistant_prompt_format,
}


def format_qa_task(
    qa_task: Dataset, *, task_format: str, answered_example: bool = False, **kwargs
) -> Dataset:
    """Format a question answering task with a specific model in mind

    At this stage, we know which model is going to perform the task, and how it want's its inputs prepared.

    When `answered_example` is `True`, the answer is included as part of self_input_text.
    This is typically used in training, or to generate solved examples for few-shot inference.

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
    with no_cache():
        # Add the two input fields
        tmp = qa_task.map(
            KNOWN_QA_TASK_FORMATS[task_format], fn_kwargs={"answered_example": answered_example}
        )
        # Remove the consumed fields
        tmp = tmp.remove_columns(["question_text", "context_texts", "contexts_headers"])
    return tmp
