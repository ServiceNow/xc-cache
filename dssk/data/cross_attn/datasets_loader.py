import random
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
import datasets
from torch.utils.data import Dataset
from typing import List, Dict, Union, Optional

from dssk.models.get_tokenizer import get_tokenizer
from dssk.data.format_qa_task import get_cross_attn_model_format
from dssk.models import infer_model_type


# Adapted from https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L341
def apply_fim_transform(
    input_tokens: List[int],
    suffix_token_id: List[int],
    prefix_token_id: List[int],
    middle_token_id: List[int],
    max_length: int,
    skip_start_n_tokens: int = 4,
    fim_spm_rate: float = 0.5,
) -> List[int]:
    input_tokens = np.array(input_tokens)

    new_tokens_length = len(suffix_token_id) + len(prefix_token_id) + len(middle_token_id)

    assert skip_start_n_tokens > 0
    assert len(input_tokens) > skip_start_n_tokens + new_tokens_length
    assert max_length > skip_start_n_tokens + new_tokens_length

    boundaries = list(
        np.random.randint(
            low=skip_start_n_tokens,
            high=min(max_length - new_tokens_length - 1, input_tokens.shape[0] - 1),
            size=2,
        )
    )
    boundaries.sort()

    prefix = input_tokens[: boundaries[0]]
    middle = input_tokens[boundaries[0] : boundaries[1]]
    suffix = input_tokens[boundaries[1] :]

    new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + new_tokens_length

    diff = new_length - max_length

    if diff > 0:  # too long
        suffix = suffix[: suffix.shape[0] - diff]

    # We repeat what was done for starcoder:
    # https://github.com/bigcode-project/Megatron-LM/blob/bd0aaba3492b441d7f186bb1159fc21e1dcd7a72/megatron/data/gpt_dataset.py#L729-L741
    # I.e. we flip a fair coin to decide the ordering.
    if np.random.binomial(1, fim_spm_rate):
        # SPM (variant 2 from FIM paper)
        input_tokens = np.concatenate(
            [prefix_token_id, suffix_token_id, suffix, middle_token_id, prefix, middle]
        ).tolist()
    else:
        # PSM
        input_tokens = np.concatenate(
            [prefix_token_id, prefix, suffix_token_id, suffix, middle_token_id, middle]
        ).tolist()

    return input_tokens


class DatasetWithContext(Dataset):
    """Indexed dataset class with context, question, answer, and context embeddings."""

    def __init__(
        self,
        train_dataset: datasets.Dataset,
        context_length: int,
        tokenizer: PreTrainedTokenizerFast,
        include_context_ids: bool,
        include_questions_on_contexts: bool,
        model_type: str,  # should be a value in {"llama", "tulu", "mistral", "gptbicode"}.
        chunked_contexts: bool,  # Split chunks of context rather than concatenating.
        return_answers: bool = False,  # This should be set only for validation data.
    ) -> None:
        """Instantiates an indexed dataset wrapping a base data source and contexts."""
        self.train_dataset = train_dataset.with_format("torch")
        self.context_length = context_length
        self.tokenizer = tokenizer

        # NOTE: Some of our models were released without FIM ids.
        # We use a sequence of tokens
        self.fim_prefix_token_ids = self.tokenizer.encode("<fim_prefix>", add_special_tokens=False)
        self.fim_middle_token_ids = self.tokenizer.encode("<fim_middle>", add_special_tokens=False)
        self.fim_suffix_token_ids = self.tokenizer.encode("<fim_suffix>", add_special_tokens=False)
        # Setting include_context_ids will make it so that we iterate over the dataset twice.
        # In the first time, we do the usual return question/answer pairs along with the context embeddings.
        # In the second time, we return the context tokens to perform causal lm on, and include the context embeddings as well.
        # If include_context_ids is not set, then only the first Q&A iteration over the data is performed, and context ids are never returned.
        self.include_context_ids = include_context_ids
        self.include_questions_on_contexts = include_questions_on_contexts
        self.chunked_contexts = chunked_contexts

        self.formatter = get_cross_attn_model_format(model_type)

        self.return_answers = return_answers

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

        formatted_example = self.formatter(
            self.train_dataset[example_idx],
            answered_example=True,
            return_context_list=self.chunked_contexts,
        )

        if use_context:
            input_str = formatted_example["useful_contexts"]
        else:
            input_str = formatted_example["self_input_str"]

        input_ids = self.tokenizer(
            input_str,
            max_length=self.context_length,
            truncation=True,
        )["input_ids"]

        if do_fim_transform:
            input_ids = apply_fim_transform(
                input_ids,
                self.fim_prefix_token_ids,
                self.fim_middle_token_ids,
                self.fim_suffix_token_ids,
                max_length=self.context_length,
            )

        # Context ids are used for embedding during training.
        if self.include_questions_on_contexts:
            context_str = formatted_example["cross_input_str_with_question"]
        else:
            context_str = formatted_example["cross_input_str"]

        if isinstance(context_str, list):
            context_input_ids = [
                self.tokenizer(
                    context,
                    max_length=self.context_length,
                    truncation=True,
                )["input_ids"]
                for context in context_str
            ]
        else:
            context_input_ids = self.tokenizer(
                context_str,
                max_length=self.context_length,
                truncation=True,
            )["input_ids"]

        processed_item = {
            "input_ids": input_ids,
            "context_input_ids": context_input_ids,
            "input_str": input_str,
            "context_str": context_str,
            "do_fim_transform": do_fim_transform,
        }

        if self.return_answers:
            no_answer_input_ids = self.tokenizer(
                formatted_example["no_answer_self_input_str"],
                max_length=self.context_length,
                truncation=True,
            )["input_ids"]
            # We drop the <eos> token of "no_answer_input_ids" since it's
            # used for generation evaluation.
            processed_item["no_answer_input_ids"] = no_answer_input_ids[:-1]

            raw_answer_input_ids = self.tokenizer(
                formatted_example["raw_answer"],
                max_length=self.context_length,
                truncation=True,
            )["input_ids"]
            processed_item["raw_answer_input_ids"] = raw_answer_input_ids

        return processed_item


class Collator:
    """Collator object mapping sequences of items from dataset instance
    into batches of Encoder tokens, decoder tokens and masks.
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

        self.return_generation_related_fields_(False)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of triplets of examples to batches of token ids, masks, and labels used for training.

        Args:
            batch (List[Dict]): List of pairs of examples.

        Returns:
            Dict[str, torch.Tensor]: Batches of decoder input tokens, encoder input tokens, and padding masks.
        """

        is_chunked_ctx = isinstance(batch[0]["context_input_ids"][0], list)

        input_ids_list = [el["input_ids"] for el in batch]

        decoder_inputs = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            max_length=self.maximum_length,
            padding="longest",
            return_tensors="pt",
        )

        processed_batch = {
            "input_ids": decoder_inputs["input_ids"],
            "attention_mask": decoder_inputs["attention_mask"],
        }

        if is_chunked_ctx:
            context_input_is_chunk_list = [el["context_input_ids"] for el in batch]

            chunk_length = len(context_input_is_chunk_list[0])

            # Converts list of list of chunks into list of chunks.
            flat_context_input_ids_list = [
                chunk for chunk_list in context_input_is_chunk_list for chunk in chunk_list
            ]

            tokenized_context_ids = self.tokenizer.pad(
                {"input_ids": flat_context_input_ids_list},
                max_length=self.maximum_length,
                padding="longest",
                return_tensors="pt",
            )

            # Back to list of chunks after tokenization.
            context_input_ids = torch.chunk(
                tokenized_context_ids["input_ids"], chunks=chunk_length
            )
            encoder_attention_mask = torch.chunk(
                tokenized_context_ids["attention_mask"],
                chunks=chunk_length,
            )

        else:
            context_input_ids_list = [el["context_input_ids"] for el in batch]

            tokenized_context_ids = self.tokenizer.pad(
                {"input_ids": context_input_ids_list},
                max_length=self.maximum_length,
                padding="longest",
                return_tensors="pt",
            )

            context_input_ids = tokenized_context_ids["input_ids"]
            encoder_attention_mask = tokenized_context_ids["attention_mask"]

        processed_batch.update(
            {
                "context_input_ids": context_input_ids,
                "encoder_attention_mask": encoder_attention_mask,
            }  # Dropout will err if we use masks of type Long
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

        # extra fields used only for extra evaluations that require generation.
        if self._return_generation_related_fields and "raw_answer_input_ids" in batch[0]:

            no_answer_input_ids_list = [el["no_answer_input_ids"] for el in batch]
            tokenized_no_answer_input_ids = self.tokenizer.pad(
                {"input_ids": no_answer_input_ids_list},
                max_length=self.maximum_length,
                padding="longest",
                return_tensors="pt",
            )
            processed_batch.update(
                {
                    "no_answer_input_ids": tokenized_no_answer_input_ids["input_ids"],
                    "no_answer_attention_mask": tokenized_no_answer_input_ids["attention_mask"],
                }
            )

            raw_answer_input_ids_list = [el["raw_answer_input_ids"] for el in batch]
            tokenized_raw_answer_input_ids = self.tokenizer.pad(
                {"input_ids": raw_answer_input_ids_list},
                max_length=self.maximum_length,
                padding="longest",
                return_tensors="pt",
            )
            processed_batch.update(
                {
                    "raw_answer_input_ids": tokenized_raw_answer_input_ids["input_ids"],
                }
            )

        return processed_batch

    def return_generation_related_fields_(self, value: bool):
        self._return_generation_related_fields = value


def data_prep(
    tokenizer_path: str,
    data_dir: str,
    context_length: int,
    chunked_contexts: bool,
    training_data_subset: str = "all",
    validation_data_subset: str = "all",
    data_cache_dir: str = None,
    include_context_ids: bool = False,
    include_questions_on_contexts: bool = True,
    model_type: Optional[str] = None,
) -> Union[List[Dataset], Dataset]:
    """Get and pre-process training dataset. This assumes data was previously prepared and context embeddings
    are available in the dataset.

    Args:
        tokenizer_path (str): Path to tokenizer.
        data_dir (str): Path to nq dataset.
        context_length (int): Maximum length of ids sequence.
        chunked_contexts (bool): Whether to return chunks of contexts as opposed to a single larger concatenation of pieces of context.
        training_data_subset (str): Optional subset corresponding to one of the datasets used to compose the training data.
        validation_data_subset (str): Optional subset corresponding to one of the datasets used to compose the training data.
        data_cache_dir (str): Optional hf path cache in case the dataset is not available in disk.
        include_context_ids (bool): Whether to include context ids in the training batch. Defaults to False.
        include_questions_on_contexts (bool): Whether to prepend questions on contexts fed to the encoder.
        model_type (Optional[str]): Which kind of model to instantiate. We currently support values in {"llama", "tulu", "mistral", "gptbicode"}.

    Returns:
        Union[List[Dataset], Dataset]: Processed datasets.
    """

    try:
        data = datasets.load_from_disk(data_dir)
    except FileNotFoundError:
        data = datasets.load_dataset(data_dir, cache_dir=data_cache_dir, use_auth_token=True)

    training_data = data["train"]
    validation_data = data["val"].shuffle().select(range(100))

    if training_data_subset.lower() != "all":
        training_data = training_data.filter(
            lambda x: x["dataset"] == training_data_subset.lower()
        )
    if validation_data_subset.lower() != "all":
        validation_data = validation_data.filter(
            lambda x: x["dataset"] == validation_data_subset.lower()
        )

    training_data = training_data.shuffle()
    # We shuffle validation data since we subsample it for evaluations that require generation.
    validation_data = validation_data.shuffle()

    tokenizer = get_tokenizer(tokenizer_path, add_eos_token=True)

    if model_type is None:
        model_type = infer_model_type(tokenizer_path)

    training_dataset = DatasetWithContext(
        training_data,
        context_length=context_length,
        tokenizer=tokenizer,
        include_context_ids=include_context_ids,
        include_questions_on_contexts=include_questions_on_contexts,
        chunked_contexts=chunked_contexts,
        return_answers=False,
        model_type=model_type,
    )
    validation_dataset = DatasetWithContext(
        validation_data,
        context_length=context_length,
        tokenizer=tokenizer,
        include_context_ids=False,
        include_questions_on_contexts=include_questions_on_contexts,
        chunked_contexts=chunked_contexts,
        return_answers=True,
        model_type=model_type,
    )

    return training_dataset, validation_dataset
