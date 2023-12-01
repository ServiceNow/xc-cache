from typing import Optional, Any

from datasets import load_dataset, Dataset

from dssk.utils.hf_datasets import no_cache


QUESTION_ID_COLUMNS = {"question_id", "question_index"}
LONG_NQ_DEDUP_ID_COLUMNS = {"long_nq_tier", "annotation_id_list", "annotation_index_list"}
TASK_COLUMNS = {"question_text", "context_texts", "contexts_headers", "answer_text"}


def empty_context_columns(d: dict[str, Any]) -> dict[str, Any]:
    return {"context_texts": [], "contexts_headers": []}


def only_gold_long_context_columns(d: dict[str, Any]) -> dict[str, Any]:
    """Provide a single context: the long answer."""
    return {
        "context_texts": [
            d["long_answer_clean"],
        ],
        "contexts_headers": [d["headers_before_long_answer"]],
    }


def newline_answer_column(d: dict[str, Any]) -> dict[str, Any]:
    """Format the (zero, one or many) short answer(s) as a newline-separated string"""
    short_answers_text = d.get("short_answers_text", ())
    yes_no_answer = d.get("yes_no_answer", -1)
    if short_answers_text:
        assert yes_no_answer == -1
        return {"answer_text": "\n".join(short_answers_text)}
    elif yes_no_answer == 1:
        return {"answer_text": "Yes"}
    elif yes_no_answer == 0:
        return {"answer_text": "No"}
    else:
        return {"answer_text": "Answer not in context."}


def deprecated_answer_column(d: dict[str, Any]) -> dict[str, Any]:
    """Drop all short answers except the first one (used to be default)"""
    short_answers_text = d.get("short_answers_text", ())
    yes_no_answer = d.get("yes_no_answer", -1)
    if short_answers_text:
        # NOTE: This code only returns the first short answer!
        assert yes_no_answer == -1
        return {"answer_text": short_answers_text[0]}
    elif yes_no_answer == 1:
        return {"answer_text": "Yes"}
    elif yes_no_answer == 0:
        return {"answer_text": "No"}
    else:
        return {"answer_text": "Answer not in context."}


KNOWN_CONTEXT_OPTIONS = {
    "empty": empty_context_columns,
    "only_gold_long": only_gold_long_context_columns,
}

KNOWN_ANSWER_OPTIONS = {"newline": newline_answer_column, "deprecated": deprecated_answer_column}


def get_qa_task(
    *,
    dataset_name: str,
    dataset_split: Optional[str],
    cache_path: str,
    context: str = "only_gold_long",
    answer: str = "newline",
    subset_size: Optional[int] = None,
    **kwargs,  # Discarded
) -> Dataset:
    """Get the correct dataset (split) and preprocess it for a specific "task"

    What constitute a "task" *excludes* any knowledge about the model that will consume it.
    This is about *what* is potentially available to the model, not *how* it is presented.
    Further processing is thus required to transform a "task" into a "model_input".

    For question answering, the following fields must be present. Extra fields can be provided
    (e.g., ids for identifying samples' source), but let's try to keep their number in check.

    - question_text: str
        A single question in a natural question format.
        - Example: "what is love".
        - Counter-example: "question: what is love".

    - context_texts: Sequence[str]
        Zero, one or many "contexts" that may or may not be relevant for answering the question. If many contexts are present, there is a semantic understanding that contexts[0] is more likely to be relevant, and that this relevance goes down with rank. Note that, for some models, this may be better achieved by formatting these these contexts in reverse order when feeding them to the model.

    - contexts_headers: Sequence[Sequence[str]]
        When present, `len(contexts_headers) == len(context_texts)`, and each entry has format [h1, h2, h3, ...]

    - answer_text: str
        A (usually short) answer.
        TODO: Some evaluation scheme may benefit from something more structured. We should think about it.

    - subset_size: Optional[int]
        The number of samples to keep as a subset of the dataset.
        For debug and/or test purpose.
    """
    # This implementation is "fake", not general at all. But it works for our evaluation purpose.
    # TODO: Come to an agreement about the interface so that we can use this "task" concept for both training/evaluation purpose.
    assert dataset_name == "long_nq_dedup"
    raw = load_dataset(f"ServiceNow/{dataset_name}", cache_dir=cache_path, split=dataset_split)
    if subset_size is not None:
        assert subset_size > 0
        raw = raw.select(range(subset_size))
    with no_cache():
        tmp = raw.map(KNOWN_ANSWER_OPTIONS[answer])
        tmp = tmp.map(KNOWN_CONTEXT_OPTIONS[context])
        tmp = tmp.select_columns(
            list(QUESTION_ID_COLUMNS | LONG_NQ_DEDUP_ID_COLUMNS | TASK_COLUMNS)
        )
        return tmp
