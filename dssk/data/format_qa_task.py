from typing import Any, Optional, Sequence, List, Callable
from datasets import Dataset

from dssk.utils.hf_datasets import update_infodict
from dssk.data.utils.pre_processors import PosContextPreProcessor


def get_cross_attn_model_format(model_type: str) -> Callable:
    if model_type == "llama":
        return cross_llama_chat_question_in_context
    if model_type == "tulu":
        return cross_uaf_question_in_context
    if model_type == "mistral":
        return cross_instruct_question_in_context
    if model_type == "gptbigcode":
        return cross_uaf_question_in_context

    raise ValueError(f"Unkown model type: {model_type}")


def get_single_context_with_trivial_strategy(d: dict[str, Any]) -> str:
    return " ".join(d["contexts_list"])


def get_context_list(d: dict[str, Any]) -> List[str]:
    useful_contexts = [
        d["contexts_list"][i]
        for i in range(len(d["contexts_list"]))
        if d["useful_contexts"][i] == 1
    ]

    useful_contexts = " ".join(useful_contexts)

    return d["contexts_list"], useful_contexts


def cross_colon_format(d: dict[str, Any], answered_example: bool, **kwargs) -> dict[str, Any]:
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


def cross_user_assistant_format(
    d: dict[str, Any], answered_example: bool, **kwargs
) -> dict[str, Any]:
    """Ancestor of cross_uaf_question_in_context"""
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


def pre_post_q_format(
    d: dict[str, Any],
    pre_q_str: str,
    post_q_str: str,
    answered_example: bool,
    return_context_list: bool,
    eos_token: str = "",
    **kwargs,
) -> dict[str, Any]:
    """For cross-attention models with a base decoder using a pre_q_str post_q_str template."""

    ctx_id_ = "<|C|>"

    answer = d.get("answer", "")
    if answered_example:
        assert answer  # Both None and "" are illegal.
    else:
        answer = ""

    if return_context_list:
        contexts_list, useful_contexts = get_context_list(d)
        cross_input_str = [
            f"{pre_q_str}{ctx_id_}{post_q_str}{context}{eos_token}" for context in contexts_list
        ]
        cross_input_str_with_question = [
            f"{pre_q_str}{d['question']}{ctx_id_}{post_q_str}{context}{eos_token}"
            for context in contexts_list
        ]

        useful_contexts = f"{pre_q_str}{ctx_id_}{post_q_str}{useful_contexts}{eos_token}"
    else:
        context = get_single_context_with_trivial_strategy(d)
        cross_input_str = f"{pre_q_str}{ctx_id_}{post_q_str}{context}{eos_token}"
        cross_input_str_with_question = (
            f"{pre_q_str}{d['question']}{ctx_id_}{post_q_str}{context}{eos_token}"
        )
        useful_contexts = f"{pre_q_str}{ctx_id_}{post_q_str}{context}{eos_token}"

    return {
        "self_input_str": f"{pre_q_str}{d['question']}{post_q_str}{answer}{eos_token}",
        "no_answer_self_input_str": f"{pre_q_str}{d['question']}{post_q_str}",
        "cross_input_str": cross_input_str,
        "cross_input_str_with_question": cross_input_str_with_question,
        "raw_answer": f"{answer}",  # Used for cross-validation
        "useful_contexts": useful_contexts,
    }


def cross_uaf_question_in_context(
    d: dict[str, Any],
    answered_example: bool,
    return_context_list: bool,
    eos_token: str = "",
    **kwargs,
) -> dict[str, Any]:
    """For cross-attention models with a base decoder using a <|user|> <|assistant|> template."""

    pre_q_str = "<|user|>\n"
    post_q_str = "\n<|assistant|>\n"

    return pre_post_q_format(
        d,
        pre_q_str=pre_q_str,
        post_q_str=post_q_str,
        answered_example=answered_example,
        eos_token=eos_token,
        return_context_list=return_context_list,
        **kwargs,
    )


def cross_instruct_question_in_context(
    d: dict[str, Any],
    answered_example: bool,
    return_context_list: bool,
    eos_token: str = "",
    **kwargs,
) -> dict[str, Any]:
    """For cross-attention models with a base decoder using a <|user|> <|assistant|> template."""

    pre_q_str = "[INST] "
    post_q_str = " [/INST] "

    return pre_post_q_format(
        d,
        pre_q_str=pre_q_str,
        post_q_str=post_q_str,
        answered_example=answered_example,
        eos_token=eos_token,
        return_context_list=return_context_list,
        **kwargs,
    )


def cross_llama_chat_question_in_context(
    d: dict[str, Any],
    answered_example: bool,
    return_context_list: bool,
    eos_token: str = "",
    **kwargs,
) -> dict[str, Any]:
    """
    Prompt for the Llama2 model.
    """

    pre_q_str = "<|user|>\n"
    post_q_str = "\n<|assistant|>\n"

    return pre_post_q_format(
        d,
        pre_q_str=pre_q_str,
        post_q_str=post_q_str,
        answered_example=answered_example,
        eos_token=eos_token,
        return_context_list=return_context_list,
        **kwargs,
    )


def tok_cut_cat_detok(
    tokenizer,
    first_prefix: str,
    other_prefix: str,
    contexts: Sequence[str],
    eos_token: str,
    max_length: int,
) -> str:
    passages = [
        (first_prefix if i == 0 else other_prefix) + context for i, context in enumerate(contexts)
    ]
    # We drop the start-of-sequence tokens.
    toked_passages = [tokenizer(passage)["input_ids"][1:] for passage in passages]
    # Figure out truncation (simple strategy, all equal length)
    # The following pertains to the `safety` value below.
    #   - Suppose we tokenize the passages, truncate them, and concatenate to get a total of N tokens.
    #   - Detokenizing this to string (plus eos) gives the desired str output.
    #   - But if we were to re-tokenize that str, we are not guaranteed to get exactly N tokens back.
    #   - Such "boundary effects" are likely to be minor though.
    #   - Instead of figuring out proper guarantees, we use an arbitrary safety margin.
    safety = 4  # This value is completely arbitrary!  ¯\_(ツ)_/¯
    per_passage_max_length = (max_length - safety) // len(toked_passages)
    tokens = [
        tok for toked_passage in toked_passages for tok in toked_passage[:per_passage_max_length]
    ]
    return tokenizer.decode(tokens) + eos_token


def cross_uaf_cut_then_cat(
    d: dict[str, Any],
    answered_example: bool,
    *,
    tokenizer,
    max_length: int,
    eos_token: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Cross <|user|> <|assistant|> format, truncate before concatenation.

    Hacky, just to try inference.
    """
    answer = d.get("answer", "")
    if answered_example:
        assert answer  # Both None and "" are illegal.
    else:
        answer = ""
    cross_input_str = tok_cut_cat_detok(
        tokenizer=tokenizer,
        first_prefix="<|user|>\n<|C|><|assistant|>\n",
        other_prefix=" ",
        contexts=d["contexts_list"],
        eos_token=eos_token,
    )
    cross_input_str_with_question = tok_cut_cat_detok(
        tokenizer=tokenizer,
        first_prefix=f"<|user|>\n{d['question']}<|C|><|assistant|>\n",
        other_prefix=" ",
        contexts=d["contexts_list"],
        eos_token=eos_token,
        max_length=max_length,
    )
    return {
        "self_input_str": f"<|user|>\n{d['question']}\n<|assistant|>\n{answer}{eos_token}",
        "cross_input_str": cross_input_str,
        "cross_input_str_with_question": cross_input_str_with_question,
    }


def system_user_assistant_prompt_format(
    d: dict[str, Any], answered_example: bool, **kwargs
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
    d: dict[str, Any], answered_example: bool, include_context: bool = True, **kwargs
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


def tulu2_prompt_format_no_context(
    d: dict[str, Any], answered_example: bool, **kwargs
) -> dict[str, Any]:
    """Same as tulu2_prompt_format, but without including the context in the prompt."""
    return tulu2_prompt_format(d, answered_example=answered_example, include_context=False)


def llama_chat_prompt_format(
    d: dict[str, Any],
    answered_example: bool,
    instruction_str: str = "Please answer the following question given the following passages. Please be brief. If you cannot answer the question, please reply with 'UNANSWERABLE'.\n",
    include_context: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Prompt for the Llama2 Chat model.

    Based on https://github.com/McGill-NLP/instruct-qa/blob/b7bfb1744f6a6b066ba9dc88b5cc9cc571c4c5e9/instruct_qa/prompt/templates.py#L103,
    with extra prompting to encourage shorter answers which can also be 'UNANSWERABLE'.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    answer = d.get("answer", "")
    if answered_example:
        assert answer  # Both None and "" are illegal.
    # Using just a space as the separator
    if include_context:
        combined_context = " ".join(context_text for context_text in d["contexts_list"]) + "\n"
    else:
        combined_context = ""
    input_str = (
        B_INST
        + " "
        + B_SYS
        + instruction_str
        + E_SYS
        + f"{combined_context}Question: {d['question']}\n"
        + E_INST
        + "\nAnswer: "
    )
    if answered_example:
        input_str = f"{input_str}{answer}"

    return {"input_str": input_str}


def llama_lora_chat_prompt_format(
    d: dict[str, Any], answered_example: bool, **kwargs
) -> dict[str, Any]:
    """
    This is the version of the prompt which was used in research-RTLM to finetune the Llama Chat model.
    """
    return llama_chat_prompt_format(
        d=d,
        answered_example=answered_example,
        instruction_str="You're an useful assistant.\n",
        **kwargs,
    )


def llama_chat_prompt_no_context_format(
    d: dict[str, Any], answered_example: bool, **kwargs
) -> dict[str, Any]:
    # The --exclude_context flag is mandatory for this prompt,
    # as its only difference from the default one is only meant for that case.
    assert kwargs.get("include_context", True) == False
    return llama_chat_prompt_format(
        d=d,
        answered_example=answered_example,
        instruction_str="Please answer the following question. Please be brief. If you cannot answer the question, please reply with 'UNANSWERABLE'.\n",
        **kwargs,
    )


def fid_format(
    d: dict[str, Any],
    answered_example: bool,
    include_title: bool,
    include_context: bool,
    max_contexts_training: Optional[int] = None,
    **kwargs,
) -> dict[str, Any]:
    # Get the passages
    if include_context:
        assert d["contexts_list"]
        if include_title:
            template = "question: {question} title: {title} context: {context}"
        else:
            # keeping the title tag, as tested models are trained with it
            template = "question: {question} title: context: {context}"
        # All the passages for all the contexts
        passages = [
            template.format(question=d["question"], title=title, context=context)
            for title, context in zip(d["titles_list"], d["contexts_list"])
        ]
        # In training, we may want a shorter list (guaranteed to have gold)
        if max_contexts_training is not None:
            assert (
                answered_example
            ), "False answered_example indicates inference, but max_contexts_training (intended for training) is provided. The context-selection mechanism uses useful_contexts to keep the gold contexts. That would be 'cheating' at inference: you should implement something that gets random contexts instead."
            passages = PosContextPreProcessor.truncate_true_list(
                passages, d["useful_contexts"], max_items=max_contexts_training
            )
    else:
        # No context: single passage that is basically the question.
        passages = [f"question: {d['question']}"]

    # Return our passages, with the answer or not
    if answered_example:
        answer = d.get("answer", "")
        assert answer  # Both None and "" are illegal.
        return {"passages": passages, "target": answer}
    else:
        return {"passages": passages}


KNOWN_QA_TASK_FORMATS = {
    "cross_colon": cross_colon_format,
    "cross_uaf": cross_user_assistant_format,
    "cross_uaf_qic": cross_uaf_question_in_context,
    "cross_instruct_qic": cross_instruct_question_in_context,
    "cross_llama_chat_qic": cross_llama_chat_question_in_context,
    "cross_uaf_cut_then_cat": cross_uaf_cut_then_cat,
    "prompt": system_user_assistant_prompt_format,
    "prompt_tulu2": tulu2_prompt_format,
    "prompt_tulu2_no_context": tulu2_prompt_format_no_context,
    "prompt_llama_chat": llama_chat_prompt_format,
    "prompt_llama_lora_chat": llama_lora_chat_prompt_format,
    "prompt_llama_chat_no_context": llama_chat_prompt_no_context_format,
    "fid": fid_format,
}


def format_qa_task(
    qa_task: Dataset,
    *,
    max_length,
    tokenizer=None,
    task_format: Optional[str] = None,
    answered_example: bool = False,
    include_title: bool = False,
    include_context: bool = True,
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
            KNOWN_QA_TASK_FORMATS[task_format],
            fn_kwargs={
                "answered_example": answered_example,
                "include_title": include_title,
                "include_context": include_context,
                "tokenizer": tokenizer,
                "max_length": max_length,
                "return_context_list": kwargs.get("return_context_list", None),
            },
            load_from_cache_file=False,
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
        {"self_input_str", "cross_input_str", "cross_input_str_with_question", "context"},
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
