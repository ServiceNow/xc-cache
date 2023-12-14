import argparse
from typing import Dict, Any, Optional
import datasets


def create_hotpotqa_splits(cache_dir: str) -> datasets.Dataset:
    hotpotqa_data = datasets.load_dataset("hotpot_qa", name="distractor", cache_dir=cache_dir)

    test_split = hotpotqa_data["validation"]

    train_val_split = hotpotqa_data["train"].train_test_split(test_size=len(test_split))

    new_splits = {
        "train": train_val_split["train"],
        "val": train_val_split["test"],
        "test": test_split,
    }

    return datasets.DatasetDict(new_splits)


class TopiOCQASplitFilter:
    def __init__(self, conversation_cutoff_idx: int, train: bool):
        self.conversation_cutoff_idx = conversation_cutoff_idx
        self.train = train

    def __call__(self, example: Dict[str, Any]) -> bool:
        return (
            self.train
            and example["Conversation_no"] <= self.conversation_cutoff_idx
            or not self.train
            and example["Conversation_no"] > self.conversation_cutoff_idx
        )


def create_topiocqa_splits(cache_dir: str, num_proc: int = 1) -> datasets.Dataset:
    topiocqa_data = datasets.load_dataset("McGill-NLP/TopiOCQA", cache_dir=cache_dir)

    test_split = topiocqa_data["validation"]

    number_of_train_conversations = len(set(topiocqa_data["train"]["Conversation_no"]))
    number_of_test_conversations = len(set(test_split["Conversation_no"]))

    assert number_of_train_conversations > number_of_test_conversations

    conversation_cutoff_id = number_of_train_conversations - number_of_test_conversations

    train_split = topiocqa_data["train"].filter(
        TopiOCQASplitFilter(conversation_cutoff_idx=conversation_cutoff_id, train=True),
        num_proc=num_proc,
    )

    val_split = topiocqa_data["train"].filter(
        TopiOCQASplitFilter(conversation_cutoff_idx=conversation_cutoff_id, train=False),
        num_proc=num_proc,
    )

    new_splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }

    return datasets.DatasetDict(new_splits)


def main(explicit_arguments: Optional[list[str]] = None) -> str:
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

    args = parser.parse_args(explicit_arguments)

    hotpotqa_split = create_hotpotqa_splits(cache_dir=args.data_cache_path)

    topiocqa_split = create_topiocqa_splits(cache_dir=args.data_cache_path, num_proc=args.num_proc)

    print("HotPotQA", hotpotqa_split)

    print("TopiOCQA", topiocqa_split)

    hotpotqa_split.push_to_hub("ServiceNow/hotpot_qa_split")
    topiocqa_split.push_to_hub("ServiceNow/TopiOCQA_split")


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
