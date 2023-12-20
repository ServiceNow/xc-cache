import textwrap
import torch
import numpy as np
from typing import List, Dict, Optional, Any
from dssk.data.utils.encoder import Encoder


def merge_arrays(A, B, A_pos_list):
    """
    Returns a new array of len |A| + |B| where
    the elements of A are at positions in A_pos_list.
    If there is a position that exceeds the number of elements in the
    merged array (overflow), then these positions will be ignored.

    Given k positions:
    If k > |A| (more positions than items in A),
      then place A in the first |A| positions of A_pos_list.
    If k < |A| (less positions than items in A),
      then position the first |A|-k items based on A_pos_list
      and concatenate the last k items of A to B.

    """

    pos_len = len(A_pos_list)
    A_len = len(A)
    B_len = len(B)
    list_len = A_len + B_len

    # If there is position overflow (pos index > available array positions)
    # then we will ignore it
    if list_len <= max(A_pos_list):
        # Remove the large positions from the list
        A_pos_list = [x for x in A_pos_list if x < list_len]
        # Update the length of position list
        pos_len = len(A_pos_list)

    # If more positions than in A,
    # use only first |A_pos_list| items in A
    if pos_len > A_len:
        A_pos_list = A_pos_list[:A_len]
    # If less positions than in A,
    # concatenate extra items in A to B
    elif pos_len < A_len:
        B = np.concatenate((B, A[pos_len:]))
        A = A[:pos_len]

    # Create a new array with the combined size
    merged_array = np.empty(list_len, dtype=A.dtype)

    # Copy elements from array A to specified positions
    merged_array[A_pos_list] = A

    # Copy elements from array B to the remaining positions
    remaining_indices = np.setdiff1d(np.arange(list_len), A_pos_list)
    merged_array[remaining_indices] = B

    return merged_array


def rearrange_list_by_pos(
    np_list: np.ndarray,
    mask: List[List[int]],
    pos_list: List[int],
    ignore_pos_overflow: bool = True,
) -> np.ndarray:
    """
    np_list: A numpy array (list of lists of text).
    mask: a 0/1 sequence, must be same dimensions as np_list.
    pos_list: a list of positions (int) with max(pos_list) < np_list.shape[1]
    ignore_pos_overflow: if True, then any position in pos_list > num of items
        in np_list will be ignored. Otherwise, an exception will be raised.
        For example, if np_list = ["a', 'b', 'c'] and pos_list = [0, 10],
        then pos 10 will be ignored and only pos 0 will have the true context.

    The function will split the np_list based on the mask, whereby
    item at position i of np_list is sent to "false_list" if there is a
    mask[i] == 0, to "true_list" otherwise.
    The function returns a new list whereby the "true" elements appear
    in positions pos_list.
    For example:
        np_list = [['a', 'b', 'c', 'd', 'e']]
        mask = [[0, 1, 0, 0, 1]]
        pos_list = [2, 3]
        returns [['a', 'c', 'b', 'e', 'd']]

    """

    # Make sure there are no duplicate indices
    len(np.unique(pos_list)) == len(pos_list)

    list_len = np_list.shape[1]

    # If either all or none of the items in the list are true,
    # then return as is
    if (list_len == np.sum(mask, axis=1)).all() or (np.sum(mask, axis=1) == 0).all():
        return np_list

    # If there is position overflow and we don't want to ignore it, raise an exception
    if list_len <= max(pos_list) and not ignore_pos_overflow:
        raise IndexError(
            f"A position index {pos_list} is out of bounds for a list of length {list_len}."
        )

    # Split the array into two lists based on the mask
    false_list = np.ma.masked_array(np_list, mask=mask)
    true_list = np.ma.masked_array(np_list, mask=~false_list.mask)

    ordered_list = [
        merge_arrays(true_list[i].compressed(), false_list[i].compressed(), pos_list)
        for i in range(false_list.shape[0])
    ]

    return ordered_list


class WikipediaPreProcessor:
    def __init__(self, encoder: Encoder, field_name_prefix: Optional[str] = "") -> None:
        self.encoder = encoder
        self.field_name_prefix = field_name_prefix

    def __call__(self, examples: Dict[str, List], rank: int) -> Dict[str, torch.Tensor]:
        title_str = examples["title"]
        text_str = examples["text"]
        title_embedding = self.encoder.encode(title_str, rank)
        text_embedding = self.encoder.encode(text_str, rank)
        examples.update(
            {
                f"{self.field_name_prefix}title_embedding": title_embedding,
                f"{self.field_name_prefix}text_embedding": text_embedding,
            }
        )

        return examples


class SquadV2PreProcessor:
    def __init__(
        self, encoder: Encoder, context_length: int, return_context_embedding_only: bool = False
    ) -> None:
        self.encoder = encoder
        self.context_length = context_length
        self.return_context_embedding_only = return_context_embedding_only

    def __call__(self, examples: Dict[str, List], rank: int) -> Dict[str, torch.Tensor]:
        context_str = [f"context: {row}" for row in examples["context"]]

        context_embedding = self.encoder.encode(context_str, rank)

        if self.return_context_embedding_only:
            return {"encoder_hidden_states": context_embedding}

        question_str = [row for row in examples["question"]]

        answer_str = []
        for row in examples["answers"]:
            try:
                answer = row["text"][0]
            except IndexError:
                answer = ""

            answer_str.append(answer)

        input_str = [f"question: {q}\nanswer: {a}" for (q, a) in zip(question_str, answer_str)]

        input_ids = self.encoder.tokenizer(
            input_str,
            max_length=self.context_length,
            truncation=True,
        )["input_ids"]

        return {"input_ids": input_ids, "encoder_hidden_states": context_embedding}


class TrainDataPreProcessor:
    # TODO: substitute/use dssk.data.text_formats
    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def __call__(self, examples: Dict[str, List], rank: int) -> Dict[str, torch.Tensor]:
        # rank is the process id. E.g., a 4-GPU processing job would have ranks in [0:4].
        # Check dssk.data.utils.encoders.Encoder

        context_str = [f"context: {row}" for row in examples["context"]]
        context_embedding = self.encoder.encode(context_str, rank)
        question_str = [row for row in examples["question"]]
        answer_str = [row for row in examples["answer"]]

        return {
            "context": context_str,
            "question": question_str,
            "answer": answer_str,
            "encoder_hidden_states": context_embedding,
        }


class PosContextPreProcessor:
    """
    The class is used to preprocess a dataset so that
    the true context(s) is always at a given positions(s).
    Useful for performing sensitivity analysis.
    It is expected to be used with the HF dataset.map() function.
    The processing can be applied to batches of size > 1,
    as long as samples in the same batch have same shape.

    The dataset to be processed is expected to have the following
    fields populated:
      - contexts_list
      - titles_list
      - context
      - useful_contexts

    The resulting dataset will alter the above four fields
    (by rearranging the list items based on position)

    """

    def __init__(self, position_list: List[int], ignore_pos_overflow: bool = True) -> None:
        """
        :position_list: list[int] of new positions where we want the true contexts to be.
        If None, then leave positions as is.
        :ignore_pos_overflow: if True, position indices that are larger than the length of
        the context list are ignored quietly. Otherwise, throw an exception.
        """
        # If we have duplicate indices, raise an exception
        if len(np.unique(position_list)) != len(position_list):
            raise IndexError(
                f"Duplicate indices!! The position list {position_list} contains duplicates"
            )

        self.position_list = position_list if position_list is None else sorted(position_list)
        self.ignore_pos_overflow = ignore_pos_overflow

    def __call__(self, examples: Dict[str, List]) -> List:
        """
        Rearrange the contexts based on the position of the true contexts
        given in self.position_list.
        Given a list of contexts and a list of true positions for each,
        rearrange each context list so that the true contexts are at the
        desired positons (self.position_list).
        """

        # If no position list given, return as is
        if self.position_list is None:
            return

        try:
            # Reorder the list of contexts/titles/masks so that
            # it reflects the new positions
            ordered_titles = rearrange_list_by_pos(
                np.array(examples["titles_list"]),
                examples["useful_contexts"],
                self.position_list,
                self.ignore_pos_overflow,
            )
            ordered_contexts = rearrange_list_by_pos(
                np.array(examples["contexts_list"]),
                examples["useful_contexts"],
                self.position_list,
                self.ignore_pos_overflow,
            )
            ordered_useful_contexts = rearrange_list_by_pos(
                np.array(examples["useful_contexts"]),
                examples["useful_contexts"],
                self.position_list,
                self.ignore_pos_overflow,
            )
        except IndexError as e:
            print(f"There was a problem with sample {examples['sample_idx']}: {e}")

        # Concatenate the contexts again based on the new ordering
        concat_ordered_contexts = [" ".join(x) for x in ordered_contexts]

        return {
            "titles_list": ordered_titles,
            "contexts_list": ordered_contexts,
            "context": concat_ordered_contexts,
            "useful_contexts": ordered_useful_contexts,
        }

    @staticmethod
    def truncate_true_list(
        item_list: List[Any], mask: List[int], max_items: int = None
    ) -> np.ndarray:
        """
        - item_list: A list or np array of any type (e.g. list of contexts),
        - mask: a 0/1 sequence, must be same dimensions as item_list.
        - max_items: The max number of list items we want to return.

        Returns a (shuffled) item_list containing :max_items number of items,
        so that the unmasked items (e.g. "true" contexts) are in the output.

        The function will split the item_list based on the mask, whereby
        item at position i of item_list is sent to "false_list"
        if mask[i] == 0, to "true_list" otherwise.
        The function then truncates the false list based on max_items,
        concatenates with true_list, and returns a shuffled list.

        For example:
            item_list = ['a', 'b', 'c', 'd', 'e']
            mask = [0, 1, 0, 0, 1]
            max_items = 3
            returns ['e', 'c', 'b']
        """

        # If no max_items specified, no truncation will occur
        if max_items is None:
            max_items = len(item_list)

        # You need to return at least one item.
        assert max_items > 0

        # Split the array into two lists based on the mask
        false_list = np.ma.masked_array(item_list, mask=mask).compressed()
        true_list = np.ma.masked_array(item_list, mask=np.logical_not(mask)).compressed()

        true_len = len(true_list)

        if max_items > true_len:
            if len(false_list) >= max_items - true_len:
                false_list = np.random.choice(
                    a=false_list, size=max_items - true_len, replace=False
                )

            # Truncate the false items
            final_list = np.concatenate([true_list, false_list])[
                :max_items
            ]  # this works even if max_items > item_len

            # Shuffle the final list, so that true items are not always at the begginning
            np.random.shuffle(final_list)
        else:
            # No false items are reported. We sample from true items instead.
            # This will also shuffle the true items
            final_list = np.random.choice(true_list, size=max_items, replace=False)

        return final_list


class SplitContextPreProcessor(PosContextPreProcessor):
    """
    The class is used to preprocess a dataset so that
    the contexts are concatenated then split into k pieces.
    There is also the option to decide the position of the true contexts in the list.
    Useful for performaing sensitivity analysis or for training models where each attention
    layer receives a different part of the context.
    """

    def __init__(
        self,
        number_of_splits: int,
        position_list: List[int] = None,
        ignore_pos_overflow: bool = True,
        split_offset: int = 5,
    ) -> None:
        """
        :number_of_splits: the number of context splits to return.
        :position_list: list[int] of new positions where we want the true contexts to be.
        If None, then leave positions as is.
        :ignore_pos_overflow: if True, position indices that are larger than the length of
        the context list are ignored quietly. Otherwise, throw an exception.
        :split_offset: the number of characters that each split is allowed to differ from each other.
        If zero, the splits are exactly the same number of characters.
        """
        if position_list is not None:
            super().__init__(position_list, ignore_pos_overflow)
        self.position_list = position_list
        self.number_of_splits = number_of_splits
        self.split_offset = split_offset

    def __call__(self, examples: Dict[str, List]) -> List:
        """
        Rearrange the contexts based on the position of the true contexts given in self.position_list.
        Given a list of contexts and a list of true positions for each, rearrange each context list
        so that the true contexts are at the desired positons (self.position_list).
        """

        if self.position_list is None:
            reordered_examples = examples
        else:
            # Reorder based on position
            reordered_examples = super().__call__(examples)

        # For each context in the example batch, compute the length of each context string split
        context_len = np.vectorize(len)(reordered_examples["context"]) // self.number_of_splits

        # Split each context into pieces, with each piece having almost the same length (+ or - 5 characters)
        split_contexts_list = [
            textwrap.wrap(
                reordered_examples["context"][i],
                width=context_len[i] + self.split_offset,
                max_lines=self.number_of_splits,
                drop_whitespace=True,
            )
            for i in range(len(reordered_examples["context"]))
        ]

        reordered_examples["split_contexts_list"] = split_contexts_list

        return reordered_examples
