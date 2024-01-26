from typing import Any, Optional, Iterable
from hashlib import sha256
import contextlib
import json

from datasets import (
    is_caching_enabled,
    disable_caching,
    enable_caching,
    Dataset,
    concatenate_datasets,
)

"""Utilities extending huggingface's datasets library.

If it could have been part of huggingface's datasets library, it probably belongs here.
This is NOT for specific datasets.
"""


@contextlib.contextmanager
def no_cache():
    was_caching_enabled = is_caching_enabled()
    try:
        disable_caching()
        yield None
    finally:
        # Put caching back as it was.
        if was_caching_enabled:
            enable_caching()


def get_infodict(ds: Dataset) -> dict[str, Any]:
    """Get information dictionary stored in a dataset's description"""
    description = ds.info.description
    if description:
        return json.loads(description)
    else:
        return {}


def set_infodict(ds: Dataset, d: dict[str, Any]) -> None:
    """Set information dictionary stored in a dataset's description

    If anything is already there, it gets overwritten.
    """
    ds.info.description = json.dumps(d)


def update_infodict(
    ds: Dataset,
    d: dict[str, Any],
    allow_overwrite: bool = False,
    source_ds: Optional[Dataset] = None,
) -> None:
    """Update information dictionary stored in a dataset's description

    New keys will be added to the existing information dictionary (if any).

    Trying to overwrite an existing key will raise an assertion error, unless
    `allow_overwrite` is set to `True`.

    If `source_ds` is provided, the original information dictionary is read from there
    instead of from `ds` (and any information dictionary in `ds` is ignored).
    """
    # Get the information dictionary before the update.
    if source_ds is None:
        source_ds = ds
    tmp_d = get_infodict(ds)

    # Check for overwrite
    if not allow_overwrite:
        assert not any(key in tmp_d for key in d.keys())

    # Do the update
    tmp_d.update(d)
    set_infodict(ds, tmp_d)


def concatenate_datasets_with_infodict(parts: list[Dataset]) -> Dataset:
    """Concatenate datasets sharing the information dictionary"""
    assert len(parts) >= 1
    d = get_infodict(parts[0])
    assert all(get_infodict(part) == d for part in parts[1:]), "Infodict mismatch."
    out = concatenate_datasets(parts)
    set_infodict(out, d)
    return out


def pseudorandom_sorted(iterable: Iterable) -> tuple:
    """Sort according to hash of str representation.

    The resulting order is pseudorandom and reproducible.
    """
    hashed_to_original = {
        sha256(str(original).encode("utf-8")).hexdigest(): original for original in iterable
    }
    sorted_hashes = sorted(hashed_to_original.keys())
    return tuple(hashed_to_original[hashed] for hashed in sorted_hashes)


def id_tuple(sample: dict[str, Any], id_columns: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(
        tuple(sample[key]) if isinstance(sample[key], list) else sample[key] for key in id_columns
    )


def subsample_deterministic(
    ds: Dataset,
    n: int,
    id_columns: tuple[str, ...],
    guaranteed_unique_column: Optional[str] = None,
) -> Dataset:
    """Deterministically subsamples a dataset

    Filters `ds` so that it contains exactly `n` rows. Running twice yields the same result.
    You need to provide column(s) `id_columns` that, together, uniquely identify a sample (ORDER MATTERS FOR DETERMINISM!).
    If `guaranteed_unique_column` is provided, we guarantee that none of those rows have the same value for that column.
    """
    id_to_index = {id_tuple(sample, id_columns): index for index, sample in enumerate(ds)}
    if guaranteed_unique_column is None:
        kept_ids = pseudorandom_sorted(id_to_index.keys())[:n]
    else:
        ids_by_guaranteed = {}
        for sample in ds:
            guaranteed = sample[guaranteed_unique_column]
            if guaranteed in ids_by_guaranteed:
                ids_by_guaranteed[guaranteed].add(id_tuple(sample, id_columns))
            else:
                ids_by_guaranteed[guaranteed] = {id_tuple(sample, id_columns)}
        assert (
            len(ids_by_guaranteed) >= n
        ), f"Cannot guarantee {n} unique values for {guaranteed_unique_column}."
        kept_ids = pseudorandom_sorted(
            pseudorandom_sorted(value)[0] for value in ids_by_guaranteed.values()
        )[:n]
    kept_indices = [id_to_index[kept] for kept in kept_ids]
    return ds.select(kept_indices, keep_in_memory=True).flatten_indices(keep_in_memory=True)


def filter_with_dict(ds: Dataset, filter: dict[str, str]) -> Dataset:
    def keep_predicate(sample) -> bool:
        return all(str(sample[key]) == value for key, value in filter.items())

    return ds.filter(keep_predicate)


def filter_with_str(ds: Dataset, filter: str) -> Dataset:
    return filter_with_dict(
        ds, {key_value.split(":")[0]: key_value.split(":")[1] for key_value in filter.split(";")}
    )
