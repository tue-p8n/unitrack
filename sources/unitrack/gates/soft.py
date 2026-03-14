"""Soft companions for gates."""

from __future__ import annotations

import dataclasses

from unitrack.data import Detections, FrameContext, Gate, Tracklets
from unitrack.states.kalman.project import mahalanobis_d2

__all__ = ["SoftMotionGate"]


@dataclasses.dataclass(frozen=True, slots=True)
class SoftMotionGate:
    """Smooth Mahalanobis: returns Gate.CostBias = chi2 / temperature."""

    mean_field: str
    cov_field: str
    temperature: float = 1.0

    def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
        """Return a CostBias gate with Mahalanobis distance / temperature."""
        del ctx
        a = getattr(cs, self.mean_field)
        cov = getattr(cs, self.cov_field)
        b = getattr(ds, self.mean_field)
        d2 = mahalanobis_d2(a, cov, b, label="SoftMotionGate")
        return Gate.CostBias(matrix=d2 / self.temperature)
