"""
Pure tracker, multi-stream/batch wrappers, and clip-level inference.

:class:`~unitrack.tracker.Tracker` is the pure step function over snapshots;
:class:`~unitrack.tracker.MultiStream` and :class:`BatchTracker` add stateful per-stream
or per-slot wrapping, and :class:`~unitrack.tracker.ClipTracker` runs the tracker over a
fixed-length clip.
"""

from __future__ import annotations

from .batch import BatchTracker
from .clip import ClipTracker
from .memory import TrackletMemory
from .multistream import (
    AutoForkOnNewKey,
    ForkPolicy,
    MultiStream,
    OrderedNoInterleaving,
)
from .tracker import StepResult, Tracker

__all__ = [
    "AutoForkOnNewKey",
    "BatchTracker",
    "ClipTracker",
    "ForkPolicy",
    "MultiStream",
    "OrderedNoInterleaving",
    "StepResult",
    "Tracker",
    "TrackletMemory",
]
