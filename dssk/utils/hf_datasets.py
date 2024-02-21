from typing import Any, Optional, Iterable, Sequence, Hashable
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
    tmp_d = get_infodict(source_ds)

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

    return ds.filter(keep_predicate).flatten_indices(keep_in_memory=True)


def filter_with_str(ds: Dataset, filter: str) -> Dataset:
    return filter_with_dict(
        ds, {key_value.split(":")[0]: key_value.split(":")[1] for key_value in filter.split(",")}
    )


def merge_duplicated_rows(
    ds: Dataset, columns_as_list: Iterable[str], columns_as_csv: Iterable[str] = ()
) -> Dataset:
    """Merge duplicated rows, with non-duplicate columns converted to lists.

    The name of columns passed in `columns_as_list` is postfixed with "_list", and their content becomes a list of the "duplicated" rows.
    The name of columns passed in `columns_as_csv` remains unchanged, and their content becomes a comma-separated string of the "duplicated" rows.
    The remaining columns are left unchanged, and their content determines whether a row is "duplicated" or not.
    """
    columns_as_list = tuple(columns_as_list)
    columns_as_csv = tuple(columns_as_csv)
    id_columns = [
        name for name in ds.column_names if name not in set(columns_as_list).union(columns_as_csv)
    ]
    id_to_indices = {}
    for index, sample in enumerate(ds):
        id_ = id_tuple(sample, id_columns)
        if id_ in id_to_indices:
            id_to_indices[id_].append(index)
        else:
            id_to_indices[id_] = [index]
    seen = set()

    def generator():
        for sample in ds:
            id_ = id_tuple(sample, id_columns)
            if id_ in seen:
                continue
            seen.add(id_)
            for name in columns_as_list:
                sample[name + "_list"] = [ds[i][name] for i in id_to_indices[id_]]
                del sample[name]
            for name in columns_as_csv:
                sample[name] = ",".join(str(ds[i][name]) for i in id_to_indices[id_])
            yield sample

    out = Dataset.from_generator(generator, keep_in_memory=True)
    mdr = get_infodict(ds).get("merge_duplicated_rows", [])
    mdr.append({"columns_as_list": list(columns_as_list), "columns_as_csv": list(columns_as_csv)})
    update_infodict(out, {"merge_duplicated_rows": mdr}, allow_overwrite=True, source_ds=ds)
    return out


def permute_like(
    target: Sequence[Hashable], current: Sequence[Hashable], *args: Sequence[Any]
) -> tuple[list[Any], ...]:
    assert set(target) == set(current)  # Matching elements
    lookup = {key: value for value, key in enumerate(current)}
    assert len(lookup) == len(target)  # No repeated elements
    permutation = tuple(lookup[key] for key in target)
    return tuple(list(seq[key] for key in permutation) for seq in (current,) + args)


def relocate_true(
    target_relative_position: float, truthness: Sequence[Any], *args: Sequence[Any]
) -> tuple[list[Any], ...]:
    """Reorder sequences such that `truthness` has its 'true' values around specified `target_relative_position`

    If `target_relative_position == 0`, all 'true' values will clump at the beginning of the sequence. If `target_relative_position == 1`, all 'true' values will clump at the end of the sequence. Intermediate values are interpolated to an `optimal_true_idx`, which may be fractional. The actual position of the 'true' values is determined by minimizing the distance between valid discrete indices and this ideal 'optimal_true_idx'.

    The 'truthness' sequence specifies the positions of those 'true' values. Any entry that python understands as 'true' (e.g., `True`, `1`, `"non-empty string"`) will be clumped around `optimal_true_idx`, and everything else (e.g., `False`, `0`, `""`) will otherwise keep its previous relative ordering. If there are no 'true' value, then no reordering is performed.

    More sequences may be passed through `args`: those sequences will undergo the same reodering as the corresponding truthness.

    The following doctest shows how to apply the same reordering to three sequences such that the first of these sequences has 'true' values clump toward the middle of the sequence. Notice that all sequences are returned as lists.
    >>> relocate_true(0.5, [True, False, True, False], [0, 1, 2, 3], "abcd")
    ([False, True, True, False], [1, 0, 2, 3], ['b', 'a', 'c', 'd'])

    The following doctest does the same, but clumping 'true' values at the end of the first sequence. Notice that, while we guarantee that the ordering is preserved for the 'false' values, there is no such guarantee for 'true' ones.
    >>> relocate_true(1.0, [True, False, True, False], [0, 1, 2, 3], "abcd")
    ([False, False, True, True], [1, 3, 2, 0], ['b', 'd', 'c', 'a'])

    The following doctest shows that no reordering is performed if there are no 'true' values.
    >>> relocate_true(0.5, [False, False, False, False], [0, 1, 2, 3], "abcd")
    ([False, False, False, False], [0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    """
    assert 0 <= target_relative_position <= 1
    assert all(len(arg) == len(truthness) for arg in args)
    # Figure out the right order
    true_in_indices = tuple(i for i, t in enumerate(truthness) if t)
    optimal_true_idx = float(target_relative_position) * (len(truthness) - 1)
    true_out_candidates = sorted((abs(i - optimal_true_idx), i) for i in range(len(truthness)))
    true_out_indices = tuple(pair[1] for pair in true_out_candidates[: len(true_in_indices)])
    false_in_indices = tuple(i for i in range(len(truthness)) if i not in true_in_indices)
    false_out_indices = tuple(i for i in range(len(truthness)) if i not in true_out_indices)
    index_map = {
        out_idx: in_idx
        for in_idx, out_idx in zip(
            true_in_indices + false_in_indices, true_out_indices + false_out_indices
        )
    }
    order = tuple(index_map[i] for i in range(len(truthness)))

    # Do the reordering
    to_reorder = (truthness,) + args
    return tuple(list(tr[i] for i in order) for tr in to_reorder)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
