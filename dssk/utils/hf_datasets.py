import contextlib

from datasets import is_caching_enabled, disable_caching, enable_caching

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
