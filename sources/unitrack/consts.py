"""
Constants used throughout the unitrack package.

Defines standard keys for TensorDicts used in tracking.
"""

from typing import Final

__all__ = ["KEY_ACTIVE", "KEY_DELTA", "KEY_FRAME", "KEY_ID", "KEY_INDEX", "KEY_START"]

KEY_INDEX: Final = "_index"
KEY_FRAME: Final = "_frame"
KEY_ID: Final = "_id"
KEY_START: Final = "_start"
KEY_ACTIVE: Final = "_active"
KEY_DELTA: Final = "_delta"
