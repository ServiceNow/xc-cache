from typing import Optional, Any

from datasets import load_dataset, load_from_disk, Dataset

from dssk.utils.hf_datasets import (
    no_cache,
    update_infodict,
    filter_with_str,
    subsample_deterministic,
)


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

SUBSAMPLE_ID_COLUMNS = {
    # dataset: column,
    None: ("sample_idx", "dataset", "contexts_list"),  # Default used if dataset not in dict.
}

SUBSAMPLE_GUARANTEED_UNIQUE_COLUMN = {
    # dataset0: column0,
    # dataset1: None,
    None: "question",  # Default used if dataset not in dict.
}


def get_qa_task(
    *,
    dataset_name: str,
    dataset_split: Optional[str],
    cache_path: str,
    task_context: Optional[str] = None,
    task_answer: Optional[str] = None,
    subset_size: Optional[int] = None,
    filter: Optional[str] = None,
    **kwargs,  # Discarded
) -> Dataset:
    """Get the correct dataset (split) and preprocess it for a specific "task"

    What constitute a "task" *excludes* any knowledge about the model that will consume it.
    This is about *what* is potentially available to the model, not *how* it is presented.
    Further processing is thus required to transform a "task" into a "model_input".

    If provided, the `task_context` and/or `task_answer` arguments specify the processing
    to be done to the corresponding parts of the task. Such processing usually requires/provides specific features/columns: the new approach is to just see if the processing
    passes, and to leave all columns there. The old contract is provided below for reference.

    If provided, the `subset_size` argument specifies the number of samples to keep as a
    subset of the dataset. This is used for debug and/or test purpose.


    *** OLD CONTRACT BELOW ***

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
    """
    if dataset_name == "repliqa-syn":
        tmp = load_from_disk("/mnt/dssk/data_rw/annotated_data/repliqa-syn_v0.0.0")["test"]
    else:
        tmp = load_dataset(f"ServiceNow/{dataset_name}", cache_dir=cache_path, split=dataset_split)

    with no_cache():
        if filter is not None:
            tmp = filter_with_str(tmp, filter)
        if subset_size is not None:
            id_columns = SUBSAMPLE_ID_COLUMNS.get(dataset_name, SUBSAMPLE_ID_COLUMNS[None])
            guaranteed_unique_column = SUBSAMPLE_GUARANTEED_UNIQUE_COLUMN.get(
                dataset_name, SUBSAMPLE_GUARANTEED_UNIQUE_COLUMN[None]
            )
            tmp = subsample_deterministic(tmp, subset_size, id_columns, guaranteed_unique_column)
        if task_answer:
            tmp = tmp.map(KNOWN_ANSWER_OPTIONS[task_answer])
        if task_context:
            tmp = tmp.map(KNOWN_CONTEXT_OPTIONS[task_context])
        update_infodict(
            tmp,
            {
                "task": {
                    "dataset_name": dataset_name,
                    "dataset_split": dataset_split,
                    "context": task_context,
                    "answer": task_answer,
                    "subset_size": subset_size,
                }
            },
        )
        return tmp
