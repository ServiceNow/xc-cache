from typing import Any, Optional
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

    If `sosource_dsurces` is provided, the original information dictionary is read from there
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
