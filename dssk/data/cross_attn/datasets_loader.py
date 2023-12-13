import random
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
import datasets
from torch.utils.data import Dataset
from typing import List, Dict, Union, Optional

from dssk.models.get_tokenizer import get_tokenizer
from dssk.data.format_qa_task import cross_user_assistant_format


def prepare_example_for_formatter(context: str, question: str, answer: str) -> Dict[str, str]:
    """Prepares the data as required by formatters."""

    return {
        "question_text": question,
        "context_texts": [
            context,
        ],
        "answer_text": answer,
    }


# Adapted from https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L341
def apply_fim_transform(
    input_tokens: List[int],
    suffix_token_id: int,
    prefix_token_id: int,
    middle_token_id: int,
    skip_start_n_tokens: int = 4,
) -> List[int]:
    assert len(input_tokens) > skip_start_n_tokens

    input_tokens = np.array(input_tokens)

    boundaries = list(
        np.random.randint(low=skip_start_n_tokens, high=input_tokens.shape[0] - 2, size=2)
    )
    boundaries.sort()

    prefix = input_tokens[: boundaries[0]]
    middle = input_tokens[boundaries[0] : boundaries[1]]
    suffix = input_tokens[boundaries[1] :]

    suffix = np.concatenate([np.array([suffix_token_id]), suffix])
    prefix = np.concatenate([np.array([prefix_token_id]), prefix])
    middle = np.concatenate([np.array([middle_token_id]), middle])

    new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0]
    diff = new_length - input_tokens.shape[0]

    if diff > 0:  # too long
        if suffix.shape[0] <= diff:
            # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
            return input_tokens
        suffix = suffix[: suffix.shape[0] - diff]

    input_tokens = np.concatenate(
        [
            suffix,
            prefix,
            middle,
        ]
    ).tolist()

    return input_tokens


class DatasetWithContextEmbedding(Dataset):
    """Indexed dataset class with context, question, answer, and context embeddings."""

    def __init__(
        self,
        train_dataset: datasets.Dataset,
        context_length: int,
        tokenizer: PreTrainedTokenizerFast,
        include_context_ids: bool,
    ) -> None:
        """Instantiates an indexed dataset wrapping a base data source and contexts."""
        self.train_dataset = train_dataset.with_format("torch")
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.fim_prefix_token_id = self.tokenizer.encode("<fim_prefix>")[0]
        self.fim_middle_token_id = self.tokenizer.encode("<fim_middle>")[0]
        self.fim_suffix_token_id = self.tokenizer.encode("<fim_suffix>")[0]
        # Setting include_context_ids will make it so that we iterate over the dataset twice.
        # In the first time, we do the usual return question/answer pairs along with the context embeddings.
        # In the second time, we return the context tokens to perform causal lm on, and include the context embeddings as well.
        # If include_context_ids is not set, then only the first Q&A iteration over the data is performed, and context ids are never returned.
        self.include_context_ids = include_context_ids

    def __len__(self) -> int:
        """Returns the length of the dataset which matches that of the base dataset.

        Returns:
            int: Dataset length.
        """
        if self.include_context_ids:
            # If include_context_ids is set, we go over the data twice.
            # We first read Q&A pairs, and then contexts.
            return 2 * len(self.train_dataset)
        else:
            return len(self.train_dataset)

    def __getitem__(self, i: int) -> List[Dict]:
        """Reads from the base datasets and returns preprocessed inputs.

        Args:
            i (int): Index to be read.

        Returns:
            List[Dict]: Processed examples.
        """

        if i >= len(self.train_dataset):
            # This branch only runs if self.include_context_ids is set.
            example_idx = i - len(self.train_dataset)
            use_context = True
            do_fim_transform = random.choice([True, False])
        else:
            # This branch runs if self.include_context_ids is not set
            # Or if self.include_context_ids is set but we are in an index for
            # the first iteration over the data.
            use_context = False
            do_fim_transform = False  # No FIM for Q&A inputs.
            example_idx = i

        context_str = self.train_dataset[example_idx]["context"]
        question_str = self.train_dataset[example_idx]["question"]
        answer_str = self.train_dataset[example_idx]["answer"]

        formatted_example = cross_user_assistant_format(
            prepare_example_for_formatter(context_str, question_str, answer_str),
            answered_example=True,
        )

        if use_context:
            input_str = formatted_example["cross_input_texts"][0][0]
        else:
            input_str = formatted_example["self_input_text"]

        input_ids = self.tokenizer(
            input_str,
            max_length=self.context_length,
            truncation=True,
        )["input_ids"]

        if do_fim_transform:
            input_ids = apply_fim_transform(
                input_ids,
                self.fim_prefix_token_id,
                self.fim_middle_token_id,
                self.fim_suffix_token_id,
            )

        # Context ids are used for embedding during training.
        context_str = formatted_example["cross_input_texts"][0][0]
        context_input_ids = self.tokenizer(
            context_str,
            max_length=self.context_length,
            truncation=True,
        )["input_ids"]

        return {
            "input_ids": input_ids,
            "context_input_ids": context_input_ids,
        }


class Collator:
    """Collator object mapping sequences of items from dataset instance
    into batches of Encoder outputs, decoder input ids and masks.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        maximum_length: int,
        **kwargs,
    ) -> None:
        """Creates instance of collator.

        Args:
            tokenizer (PreTrainedTokenizerFast): Tokenizer used to convert decoder inputs to token ids.
            maximum_length (int): Truncating length of token sequences.
        """

        self.tokenizer = tokenizer
        self.maximum_length = maximum_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of triplets of examples to batches of token ids, masks, and labels used for training.
        The first two elements in a triplet correspond to neighbor chunks from the same file. The third
        element corresponds to a chunk from a random file.

        Args:
            batch (List[Dict]): List of pairs of examples.

        Returns:
            Dict[str, torch.Tensor]: Batches of decoder input tokens, encoder hidden states, and padding masks.
        """

        input_ids_list = [el["input_ids"] for el in batch]

        processed_batch = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            max_length=self.maximum_length,
            padding="longest",
            return_tensors="pt",
        )

        context_input_ids_list = [el["context_input_ids"] for el in batch]

        tokenized_context_ids = self.tokenizer.pad(
            {"input_ids": context_input_ids_list},
            max_length=self.maximum_length,
            padding="longest",
            return_tensors="pt",
        )

        processed_batch.update(
            {
                "context_input_ids": tokenized_context_ids["input_ids"],
                "encoder_attention_mask": tokenized_context_ids["attention_mask"].float(),
            }  # Dropout will err if we use types of type Long
        )

        # We repeat what is done in hugging face and labels are a simple clone of input ids
        # and shifiting happens inside the model's Forward during loss computation.
        # C.f. https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L745-L748
        # C.f. https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L803-L806
        labels = processed_batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            # Padding tokens for labels are replaced by -100 so they can be ignored
            # during training.
            labels[labels == self.tokenizer.pad_token_id] = -100
        processed_batch["labels"] = labels

        return processed_batch


def data_prep(
    tokenizer_path: str,
    data_dir: str,
    context_length: int,
    data_cache_dir: str = None,
    include_context_ids: Optional[bool] = False,
) -> Union[List[Dataset], Dataset]:
    """Get and pre-process training dataset. This assumes data was previously prepared and context embeddings
    are available in the dataset.

    Args:
        tokenizer_path (str): Path to tokenizer.
        data_dir (str): Path to nq dataset.
        context_length (int): Maximum length of ids sequence.
        data_cache_dir (str): Optional hf path cache in case the dataset is not available in disk.
        include_context_ids (Optional[bool]): Whether to include context ids in the training batch. Defaults to False.

    Returns:
        Union[List[Dataset], Dataset]: Processed datasets.
    """

    try:
        data = datasets.load_from_disk(data_dir)
    except FileNotFoundError:
        data = datasets.load_dataset(data_dir, cache_dir=data_cache_dir)

    training_data = data["train"]
    validation_data = data["val"]

    tokenizer = get_tokenizer(
        tokenizer_path,
    )

    training_dataset = DatasetWithContextEmbedding(
        training_data,
        context_length=context_length,
        tokenizer=tokenizer,
        include_context_ids=include_context_ids,
    )
    validation_dataset = DatasetWithContextEmbedding(
        validation_data,
        context_length=context_length,
        tokenizer=tokenizer,
        include_context_ids=False,
    )

    return training_dataset, validation_dataset
