import random
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
import datasets
from torch.utils.data import Dataset
from typing import List, Dict, Union, Optional

from dssk.models.get_tokenizer import get_tokenizer

ENCODER_EMBEDDING_PADDING_VALUE = -100.0


def augment_qa_str(input_str: str, question_str: str, answer_str: str) -> str:
    _augmentation_mode = random.choice(["none", "repeat_answer", "repeat_question", "repeat_both"])

    if _augmentation_mode == "none":
        to_append_str = ""
    elif _augmentation_mode == "repeat_answer":
        to_append_str = f" \n repeat answer: \n {answer_str}"
    elif _augmentation_mode == "repeat_question":
        to_append_str = f" \n repeat question: \n {question_str}"
    elif _augmentation_mode == "repeat_both":
        to_append_str = f" \n repeat both question and answer: \n {input_str}"

    return input_str + to_append_str


def augment_ctx_str(context_str: str) -> str:
    _augmentation_mode = random.choice(["none", "repeat_context"])

    if _augmentation_mode == "none":
        to_append_str = ""
    elif _augmentation_mode == "repeat_context":
        to_append_str = f" \n repeat context: \n {context_str}"

    return context_str + to_append_str


# Adapted from https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L341
def apply_fim_transform(
    input_tokens: List[int], suffix_token_id: int, prefix_token_id: int, middle_token_id: int
) -> List[int]:
    input_tokens = np.array(input_tokens)

    boundaries = list(np.random.randint(low=1, high=input_tokens.shape[0] - 2, size=2))
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
        nq_dataset: datasets.Dataset,
        context_length: int,
        tokenizer: PreTrainedTokenizerFast,
        include_context_ids: bool,
        perform_augmentations: bool,
    ) -> None:
        """Instantiates an indexed dataset wrapping a base data source and contexts."""
        self.nq_dataset = nq_dataset.with_format("torch")
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
        self.perform_augmentations = perform_augmentations

    def __len__(self) -> int:
        """Returns the length of the dataset which matches that of the base dataset.

        Returns:
            int: Dataset length.
        """
        if self.include_context_ids:
            # If include_context_ids is set, we go over the data twice.
            # We first read Q&A pairs, and then contexts.
            return 2 * len(self.nq_dataset)
        else:
            return len(self.nq_dataset)

    def __getitem__(self, i: int) -> List[Dict]:
        """Reads from the base datasets and returns preprocessed inputs.

        Args:
            i (int): Index to be read.

        Returns:
            List[Dict]: Processed examples.
        """

        if i >= len(self.nq_dataset):
            # This branch only runs if self.include_context_ids is set.
            example = self.nq_dataset[i - len(self.nq_dataset)]
            do_fim_transform = random.choice([True, False])
            input_str = f"context: {example['context']}"
            if self.perform_augmentations and not do_fim_transform:
                input_str = augment_ctx_str(input_str)
        else:
            # This branch runs if self.include_context_ids is not set
            # Or if self.include_context_ids is set but we are in an index for
            # the first iteration over the data.
            do_fim_transform = False  # No FIM for Q&A inputs.
            example = self.nq_dataset[i]
            question_str = example["question"]
            answer_str = example["answer"]
            input_str = f"question: {question_str} \n answer: {answer_str}"

            if self.perform_augmentations:
                input_str = augment_qa_str(input_str, question_str, answer_str)

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

        context_embedding = example["encoder_hidden_states"]

        return {
            "input_ids": input_ids,
            "encoder_hidden_states": context_embedding,
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

        # Besides padding, this will also truncate to sequences to the max. length.
        encoder_hidden_states_and_masks = self._pad_encoder_hidden_states(
            [el["encoder_hidden_states"] for el in batch], truncate=True
        )

        processed_batch.update(encoder_hidden_states_and_masks)

        # We repeate what is done in hugging face and labels are a simple clone of input ids
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

    def _pad_encoder_hidden_states(
        self, encoder_hidden_states: List[torch.Tensor], truncate: Optional[bool] = True
    ) -> Dict[str, torch.Tensor]:
        """
        Pads encoder hidden states with dummy vectors and creates padding masks.

        Args:
                encoder_hidden_states (List[torch.Tensor]): List of tensors with encoder hidden states.
                truncate (Optional[bool]): Whether to truncate sequences of hidden states to max. length.

            Returns:
                Dict[str, torch.Tensor]: Batches of encoder hidden states, and padding masks.
        """

        # Creates tensors out of nested lists of features
        encoder_hidden_states_tensors = [torch.Tensor(el) for el in encoder_hidden_states]

        maximum_batch_length = max([el.size(0) for el in encoder_hidden_states_tensors])
        if truncate:
            maximum_batch_length = min(maximum_batch_length, self.maximum_length)

        padding_vectors = ENCODER_EMBEDDING_PADDING_VALUE * torch.ones(
            maximum_batch_length, encoder_hidden_states_tensors[0].size(-1)
        )

        hidden_states_list, attn_masks_list = [], []

        for hidden_states in encoder_hidden_states_tensors:
            padding_mask = torch.ones(hidden_states.size(0))
            amount_padding_needed = maximum_batch_length - hidden_states.size(0)

            if amount_padding_needed > 0:
                hidden_states = torch.cat([hidden_states, padding_vectors[:amount_padding_needed]])
                padding_mask = torch.cat([padding_mask, torch.zeros(amount_padding_needed)])
            else:
                hidden_states = hidden_states[:maximum_batch_length, ...]
                padding_mask = padding_mask[:maximum_batch_length, ...]

            # We create batch dimension to enable concatenation.
            hidden_states_list.append(hidden_states[None, ...])
            attn_masks_list.append(padding_mask[None, ...])

        return {
            "encoder_hidden_states": torch.cat(hidden_states_list),
            "encoder_attention_mask": torch.cat(attn_masks_list),
        }


def nq_prep(
    tokenizer_path: str,
    data_dir: str,
    context_length: int,
    do_repetition_augmentations: bool,
    include_context_ids: Optional[bool] = False,
) -> Union[List[Dataset], Dataset]:
    """Get and pre-process NQ dataset. This assumes data was previously prepared and context embeddings
    are available in the dataset.

    Args:
        tokenizer_path (str): Path to tokenizer.
        data_dir (str): Path to nq dataset.
        context_length (int): Maximum length of ids sequence.
        do_repetition_augmentations (bool): Whether to perform repetition augmentations.
        include_context_ids (Optional[bool]): Whether to include context ids in the training batch. Defaults to False.

    Returns:
        Union[List[Dataset], Dataset]: Processed datasets.
    """

    nq_data = datasets.load_from_disk(data_dir)

    nq_training_data = nq_data["train"]
    nq_validation_data = nq_data["val"]

    tokenizer = get_tokenizer(
        tokenizer_path,
    )

    training_dataset = DatasetWithContextEmbedding(
        nq_training_data,
        context_length=context_length,
        tokenizer=tokenizer,
        perform_augmentations=do_repetition_augmentations,
        include_context_ids=include_context_ids,
    )
    validation_dataset = DatasetWithContextEmbedding(
        nq_validation_data,
        context_length=context_length,
        tokenizer=tokenizer,
        perform_augmentations=False,
        include_context_ids=False,
    )

    return training_dataset, validation_dataset
