"""
Tracklet lifecycle status enum.

Integer values are persisted into snapshot fields; do not renumber casually.
"""

from __future__ import annotations

import enum

__all__ = ["TrackletStatus"]


class TrackletStatus(enum.IntEnum):
    """Lifecycle state of a tracklet."""

    Tentative = 0
    Active = 1
    Lost = 2
    Removed = 3
