from abc import ABC, abstractmethod
import os
from typing import Any, Optional, Callable

from datasets import Dataset, load_from_disk

from dssk.utils.hf_datasets import update_infodict


class AbstractLMInterface(ABC):
    """Interface through which we can interact with a broad variety of language models."""

    # TODO: Add property "model_descriptor" giving name/checkpoint

    @property
    def model_info(self) -> dict[str, Any]:
        """Information about this model, how it was train, etc.

        Anything that could affect performances and/or be relevant in evaluation may appear here.

        This default implementation should be overridden.
        TODO: Implement this method in all our interfaces, then consider making this an @abstractmethod.
        """
        return {"ModelInfoNotImplementedInInterface": "You should implement it :)"}

    @abstractmethod
    def __call__(self, sample: dict[str, Any], **gen_args) -> dict[str, Any]:
        """Use the model to add output_text to a sample (for use with HF's map)

        The returned dict minimally contains the following key.
        - output_text: str
            What the model generates.

        The following keys in `sample` have specific meanings.
        - self_input_text: str
            This is the main input of the decoder, to be auto-regressively extended to produce `text_output`.
        - cross_input_texts: Sequence[Sequence[str]]
            Optional inputs to cross-attend to. The outer sequence typically lists different "contexts", the inner sequence lists "chunks". When there is no chunking, the inner sequence always has length 1. If there is also only one context to cross attend to, then you're probably looking for `sample["cross_input_texts"][0][0]`.

        In some cases (e.g., chunking), it may makes more sense to provide self_input_ids and/or cross_input_ids: when such *_ids are provided, they should take precedence over *_text.
        """

    def process_dataset(
        self,
        dataset: Dataset,
        named_cache_path: Optional[str] = None,
        desc: Optional[str] = None,
        post_process: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> Dataset:
        """Call the model on each sample of a dataset, and return results as an augmented dataset.

        If `named_cache_path` is provided, the output dataset is saved (with
        a proper `save_to_disk`, not through HuggingFace's default caching) if
        it didn't exist already. If `named_cache_path` already exists, the
        dataset processing is skipped and this cache is loaded instead.
        """
        # If the named cache exists, return it.
        if named_cache_path is not None and os.path.exists(
            os.path.join(named_cache_path, "dataset_info.json")
        ):
            return load_from_disk(named_cache_path)

        # Ok, no named cache, so we need to do the processing.
        # We set `load_from_cache_file` and `new_fingerprint` to prevent HF's
        # fingerprint computation from making a temporary copy of the model.
        out = dataset.map(
            self,
            load_from_cache_file=False,
            new_fingerprint="version_1",
            desc=desc,
        )
        # Add model information dictionary
        update_infodict(out, {"model": self.model_info})
        # Post processing
        if post_process:
            out = post_process(out)
        # Save and/or return output.
        if named_cache_path is not None:
            out.save_to_disk(named_cache_path)
        return out
