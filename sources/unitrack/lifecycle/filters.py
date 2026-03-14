"""Predicate functions used by the pipeline.Filter combinator."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import Tracklets

from .status import TrackletStatus

__all__ = ["MaxAgeFilter", "StatusFilter"]


@dataclasses.dataclass(frozen=True, slots=True)
class MaxAgeFilter:
    """Keep tracklets whose ``time_since_update`` is ``<= max_age``."""

    max_age: int

    def __call__(self, cs: Tracklets) -> torch.Tensor:
        """Return a bool mask, True where ``time_since_update <= max_age``."""
        return cs.time_since_update <= self.max_age


@dataclasses.dataclass(frozen=True, slots=True)
class StatusFilter:
    """Keep tracklets whose ``status`` is in ``allowed``."""

    allowed: tuple[TrackletStatus, ...]

    def __init__(self, *allowed: TrackletStatus) -> None:
        """Accept one or more allowed TrackletStatus values."""
        if not allowed:
            msg = "StatusFilter requires at least one allowed status"
            raise ValueError(msg)
        # frozen=True forbids regular setattr; ``object.__setattr__`` is the
        # documented escape hatch for assigning fields from a custom __init__.
        object.__setattr__(self, "allowed", tuple(allowed))

    def __call__(self, cs: Tracklets) -> torch.Tensor:
        """Return a bool mask, True where ``status`` is in ``allowed``."""
        keep = torch.zeros_like(cs.status, dtype=torch.bool)
        for s in self.allowed:
            keep |= cs.status == int(s)
        return keep
