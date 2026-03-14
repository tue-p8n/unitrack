"""Spatial-distance gates."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import Detections, FrameContext, Gate, Tracklets

__all__ = ["SpatialGate2D", "SpatialGate3D"]


def _pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2.0)


def _check_dims(field: str, a: torch.Tensor, b: torch.Tensor, needed: int) -> None:
    if a.shape[-1] < needed or b.shape[-1] < needed:
        msg = (
            f"spatial gate on field {field!r} needs at least {needed} components; "
            f"got cs={a.shape[-1]}, ds={b.shape[-1]}"
        )
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialGate2D:
    """2-D spatial gate — keep pairs within max_dist Euclidean distance."""

    field: str
    max_dist: float

    def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
        """Return a PerPair gate based on 2-D pairwise L2 distance."""
        del ctx
        a = getattr(cs, self.field)
        b = getattr(ds, self.field)
        _check_dims(self.field, a, b, 2)
        return Gate.PerPair(mask=_pairwise_l2(a[:, :2], b[:, :2]) <= self.max_dist)


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialGate3D:
    """3-D spatial gate — keep pairs within max_dist Euclidean distance."""

    field: str
    max_dist: float

    def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
        """Return a PerPair gate based on 3-D pairwise L2 distance."""
        del ctx
        a = getattr(cs, self.field)
        b = getattr(ds, self.field)
        _check_dims(self.field, a, b, 3)
        return Gate.PerPair(mask=_pairwise_l2(a[:, :3], b[:, :3]) <= self.max_dist)
