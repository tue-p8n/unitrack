"""
Typed records carried by the stage tree.

The data layer defines :class:`~unitrack.data.Tracklets`,
:class:`~unitrack.data.Detections`, :class:`FrameContext`,
:class:`~unitrack.data.CostExpression`, :class:`Gate`, and
:class:`~unitrack.data.MatchOutcome`, together with clip-aware wrappers for sequence
inference.
"""

from __future__ import annotations

from .clip import (
    ClipDetections,
    ClipFrameContext,
    ClipMatchOutcome,
    ClipTracklets,
    StackedClipMatch,
)
from .cost import CostExpression
from .detections import Detections
from .frame import FrameContext
from .gate import Gate
from .match import MatchOutcome
from .tensor_spec import TensorSpec
from .tracklets import Tracklets

__all__ = [
    "ClipDetections",
    "ClipFrameContext",
    "ClipMatchOutcome",
    "ClipTracklets",
    "CostExpression",
    "Detections",
    "FrameContext",
    "Gate",
    "MatchOutcome",
    "StackedClipMatch",
    "TensorSpec",
    "Tracklets",
]
