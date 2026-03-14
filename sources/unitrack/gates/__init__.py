"""Gates layer."""

from __future__ import annotations

from .motion import MotionGate
from .simple import ClassGate, NoneGate, ScoreGate
from .soft import SoftMotionGate
from .spatial import SpatialGate2D, SpatialGate3D

__all__ = [
    "ClassGate",
    "MotionGate",
    "NoneGate",
    "ScoreGate",
    "SoftMotionGate",
    "SpatialGate2D",
    "SpatialGate3D",
]
