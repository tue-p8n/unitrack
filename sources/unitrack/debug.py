"""Simple system to debug tracking modules via process output messages."""

import functools
import os

__all__ = ["check_debug_enabled"]


@functools.cache
def check_debug_enabled():
    """
    Check whether debugging is enabled.

    Reads the environment variable ``UNITRACK_DEBUG``.
    """
    return os.getenv("UNITRACK_DEBUG") is not None