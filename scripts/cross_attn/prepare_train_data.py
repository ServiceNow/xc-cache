import argparse
import datasets
from typing import List, Dict, Set, Any, Optional
import string
from random import Random


_RNG_SEED = 12345
_NO_ANSWER_STR = "UNANSWERABLE"


def process_str(s: str) -> str:
    s = s.lower()
    # Strip punctuation
    return s.translate(str.maketrans("", "", string.punctuation))


def get_nq_open_question_set(cache_dir: str) -> Set[str]:
    nq_open_data = datasets.load_dataset("nq_open", cache_dir=cache_dir)

    sentences = set()

    for split in nq_open_data.keys():
        split_sentences = set()
        for row in nq_open_data[split]:
            split_sentences.add(process_str(row["question"]))

        sentences = set.union(sentences, split_sentences)

    return sentences


class NQFilter:
    def __init__(self, nq_open_questions: Set[str]):
        self.nq_open_questions = nq_open_questions

    def __call__(self, example: Dict[str, Any]) -> bool:
        return process_str(example["question"]) in self.nq_open_questions


def ms_marco_filter(example: Dict[str, Any]) -> bool:
    return len(example["answer"]) > 0


def nq_pre_processor(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    context_str = example["long_answer_clean"]
    title_str = example["document_title"]
    question_str = example["question_text"]

    if len(example["short_answers_text"]) > 0:
        answer_str = example["short_answers_text"][0]
    elif example["yes_no_answer"] == 1:
        answer_str = "Yes"
    elif example["yes_no_answer"] == 0:
        answer_str = "No"
    else:
        answer_str = _NO_ANSWER_STR

    return {
        "context": context_str,
        "contexts_list": [context_str],
        "titles_list": [title_str],
        "useful_contexts": [1],
        "question": question_str,
        "answer": answer_str,
        "sample_idx": idx,
        "dataset": "nq",
    }


def topiocqa_pre_processor(example: Dict[str, Any], idx: int) -> Dict[str, str]:
    context_str = example["Gold_passage"]["text"]
    title_str = example["Gold_passage"]["title"]
    question_str = example["Question"]
    answer_str = example["Answer"]

    return {
        "context": context_str,
        "contexts_list": [context_str],
        "titles_list": [title_str],
        "useful_contexts": [1],
        "question": question_str,
        "answer": answer_str,
        "sample_idx": idx,
        "dataset": "topiocqa",
    }


def hotpotqa_pre_processor(
    example: Dict[str, Any], idx: int, rng: Random, shuffle_idx: int
) -> Dict[str, str]:
    # shuffle_idx is there so that Hugging Face does not reuse its cache for each invocation

    raw_contexts_list = example["context"]["sentences"]
    titles_list = example["context"]["title"]
    useful_contexts = [
        1 if title in example["supporting_facts"]["title"] else 0 for title in titles_list
    ]
    question_str = example["question"]
    answer_str = example["answer"]

    # Each paragraph is split into sentences with the post-period space still present
    # at the beginning of each sentences but the firsts.
    contexts_list = ["".join(paragraph) for paragraph in raw_contexts_list]

    # Shuffle the contexts, in case there are biases as to where the relevant ones are in the data.
    permutation = list(range(len(contexts_list)))
    rng.shuffle(permutation)
    shuffled_context_lists = [contexts_list[p] for p in permutation]
    shuffled_titles_list = [titles_list[p] for p in permutation]
    shuffled_useful_contexts = [useful_contexts[p] for p in permutation]

    # We concatenate the paragraphs adding a single space between each.
    # An alternative solution would be to add a line-break between paragraphs instead.
    context_str = " ".join(shuffled_context_lists)

    return {
        "context": context_str,
        "contexts_list": shuffled_context_lists,
        "titles_list": shuffled_titles_list,
        "useful_contexts": shuffled_useful_contexts,
        "question": question_str,
        "answer": answer_str,
        "sample_idx": idx,
        "dataset": "hotpotqa",
    }


def squad_pre_processor(example: Dict[str, Any], idx: int) -> Dict[str, str]:
    context_str = example["context"]
    title_str = example["title"]
    question_str = example["question"]
    try:
        # Sometimes squad has multiple questions so we get the first one only
        answer_str = example["answers"]["text"][0]
    except IndexError:
        # Sometimes there's no answer, so we assume the question can't be answered
        # with the given context.
        answer_str = _NO_ANSWER_STR

    return {
        "context": context_str,
        "contexts_list": [context_str],
        "titles_list": [title_str],
        "useful_contexts": [1],
        "question": question_str,
        "answer": answer_str,
        "sample_idx": idx,
        "dataset": "squad_v2",
    }


def msmarco_pre_processor(example: Dict[str, Any], idx: int) -> Dict[str, str]:
    contexts_list = example["passages"]["passage_text"]
    context_str = " ".join(contexts_list)
    # We don't have titles for MS Marco, only URLs
    titles_list = ["" for _ in range(len(contexts_list))]
    useful_contexts = example["passages"]["is_selected"]
    question_str = example["query"]
    # Sometimes squad has multiple questions
    try:
        answer_str = example["answers"][0]
    except IndexError:
        # Empty strings will be filtered out.
        answer_str = ""

    if "no answer present" in answer_str.lower():
        answer_str = _NO_ANSWER_STR

    return {
        "context": context_str,
        "contexts_list": contexts_list,
        "titles_list": titles_list,
        "useful_contexts": useful_contexts,
        "question": question_str,
        "answer": answer_str,
        "sample_idx": idx,
        "dataset": "msmarco",
    }


def get_filter_and_preproc_nq(
    nq_open_questions: Set[str], cache_dir: str, num_proc: int = 1
) -> datasets.Dataset:
    nq_data = datasets.load_dataset("ServiceNow/long_nq", cache_dir=cache_dir)

    processed_datasets_dict = {}

    for split in nq_data.keys():
        processed_nq_split = nq_data[split].map(
            nq_pre_processor,
            remove_columns=nq_data[split].column_names,
            batched=False,
            num_proc=num_proc,
            with_indices=True,
        )
        filtered_processed_nq_split = processed_nq_split.filter(NQFilter(nq_open_questions))

        processed_datasets_dict[split] = filtered_processed_nq_split

    processed_nq = datasets.DatasetDict(processed_datasets_dict)

    return processed_nq


def get_and_preproc_topiocqa(cache_dir: str, num_proc: int = 1) -> datasets.Dataset:
    topiocqa_data = datasets.load_dataset("ServiceNow/TopiOCQA_split", cache_dir=cache_dir)

    processed_datasets_dict = {}

    for split in topiocqa_data.keys():
        processed_topiocqa_split = topiocqa_data[split].map(
            topiocqa_pre_processor,
            remove_columns=topiocqa_data[split].column_names,
            batched=False,
            num_proc=num_proc,
            with_indices=True,
        )

        processed_datasets_dict[split] = processed_topiocqa_split

    processed_topiocqa = datasets.DatasetDict(processed_datasets_dict)

    return processed_topiocqa


def get_and_preproc_hotpotqa(
    num_shuffles: int, cache_dir: str, num_proc: int = 1
) -> datasets.Dataset:
    hotpotqa_data = datasets.load_dataset("ServiceNow/hotpot_qa_split", cache_dir=cache_dir)

    rng = Random(_RNG_SEED)

    processed_datasets_dict = {}
    for split in hotpotqa_data.keys():
        processed_hotpotqa_split = datasets.concatenate_datasets(
            [
                hotpotqa_data[split].map(
                    hotpotqa_pre_processor,
                    remove_columns=hotpotqa_data[split].column_names,
                    batched=False,
                    num_proc=num_proc,
                    with_indices=True,
                    fn_kwargs={"rng": rng, "shuffle_idx": shuffle_idx},
                )
                for shuffle_idx in range(num_shuffles)
            ]
        )

        processed_datasets_dict[split] = processed_hotpotqa_split

    processed_hotpotqa = datasets.DatasetDict(processed_datasets_dict)

    return processed_hotpotqa


def get_and_preproc_squad(cache_dir: str, num_proc: int = 1) -> datasets.Dataset:
    squad_data = datasets.load_dataset("squad_v2", cache_dir=cache_dir)

    processed_datasets_dict = {}

    for split in squad_data.keys():
        processed_squad_split = squad_data[split].map(
            squad_pre_processor,
            remove_columns=squad_data[split].column_names,
            batched=False,
            num_proc=num_proc,
            with_indices=True,
        )

        if split == "validation":
            # Match NQ's split naming.
            split = "val"
        processed_datasets_dict[split] = processed_squad_split

    processed_squad = datasets.DatasetDict(processed_datasets_dict)

    return processed_squad


def get_and_preproc_msmarco(cache_dir: str, num_proc: int = 1) -> datasets.Dataset:
    msmarco_data = datasets.load_dataset("ms_marco", "v2.1", cache_dir=cache_dir)

    processed_datasets_dict = {}

    for split in msmarco_data.keys():
        processed_msmarco_split = msmarco_data[split].map(
            msmarco_pre_processor,
            remove_columns=msmarco_data[split].column_names,
            batched=False,
            num_proc=num_proc,
            with_indices=True,
        )

        filtered_processed_msmarco_split = processed_msmarco_split.filter(ms_marco_filter)

        if split == "validation":
            # Match NQ's split naming.
            split = "val"
        processed_datasets_dict[split] = filtered_processed_msmarco_split

    processed_msmarco = datasets.DatasetDict(processed_datasets_dict)

    return processed_msmarco


def stack_datasets(dataset_list: List[datasets.Dataset]) -> datasets.Dataset:
    # It is OK if the "test" split is missing for a dataset
    splits = ["train", "val", "test"]

    datasets_dict = {}

    for split in splits:
        datasets_dict[split] = datasets.concatenate_datasets(
            [data[split] for data in dataset_list if split in data]
        )

    return datasets.DatasetDict(datasets_dict)


def main(explicit_arguments: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to run in parallel for map. Must be equal to the number of gpus on the node.",
    )
    parser.add_argument(
        "--data_cache_path",
        default=None,
        help="Path where to cache data.",
    )
    parser.add_argument(
        "--hotpotqa_num_shuffles",
        type=int,
        default=3,
        help="Create that many copies of the HotpotQA dataset, each with their own independently shuffled contexts.",
    )
    parser.add_argument(
        "--hub_push_path",
        help="Optional id to push the processed data to hugging face datasets.",
    )

    args = parser.parse_args(explicit_arguments)

    nq_open_questions = get_nq_open_question_set(cache_dir=args.data_cache_path)

    nq_dataset = get_filter_and_preproc_nq(
        nq_open_questions=nq_open_questions, cache_dir=args.data_cache_path, num_proc=args.num_proc
    )

    print("Natural Questions", nq_dataset)

    topiocqa_dataset = get_and_preproc_topiocqa(
        cache_dir=args.data_cache_path, num_proc=args.num_proc
    )

    print("TopioCQA", topiocqa_dataset)

    hotpotqa_dataset = get_and_preproc_hotpotqa(
        num_shuffles=args.hotpotqa_num_shuffles,
        cache_dir=args.data_cache_path,
        num_proc=args.num_proc,
    )

    print("HotpotQA", hotpotqa_dataset)

    squad_dataset = get_and_preproc_squad(
        cache_dir=args.data_cache_path,
        num_proc=args.num_proc,
    )

    print("SquadV2", squad_dataset)

    msmarco_dataset = get_and_preproc_msmarco(
        cache_dir=args.data_cache_path,
        num_proc=args.num_proc,
    )

    print("MSMarco", msmarco_dataset)

    complete_dataset = stack_datasets(
        [nq_dataset, topiocqa_dataset, hotpotqa_dataset, squad_dataset, msmarco_dataset]
    )

    print("All data", complete_dataset)

    if args.hub_push_path is not None:
        complete_dataset.push_to_hub(args.hub_push_path)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
