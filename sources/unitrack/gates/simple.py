"""Built-in gates: NoneGate, ClassGate, ScoreGate."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import Detections, FrameContext, Gate, Tracklets

__all__ = ["ClassGate", "NoneGate", "ScoreGate"]


@dataclasses.dataclass(frozen=True, slots=True)
class NoneGate:
    """Identity gate — always-True PerPair mask."""

    def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
        """Return a PerPair gate with all-True mask of shape (N, M)."""
        del ctx
        n, m = cs.batch_size[0], ds.batch_size[0]
        mask = torch.ones((n, m), dtype=torch.bool, device=cs.id.device)
        return Gate.PerPair(mask=mask)


@dataclasses.dataclass(frozen=True, slots=True)
class ClassGate:
    """Outer-equal class match — Gate.PerPair."""

    field: str

    def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
        """Return a PerPair gate where mask[i,j] = (cs[i].field == ds[j].field)."""
        del ctx
        a = getattr(cs, self.field)
        b = getattr(ds, self.field)
        return Gate.PerPair(mask=a[:, None] == b[None, :])


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreGate:
    """Per-detection score threshold — Gate.PerDs."""

    field: str
    threshold: float

    def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
        """Return a PerDs gate keeping detections with score >= threshold."""
        del cs, ctx
        return Gate.PerDs(mask=getattr(ds, self.field) >= self.threshold)
