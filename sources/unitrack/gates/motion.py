"""Mahalanobis (χ²) motion gate using Kalman state covariance."""

from __future__ import annotations

import dataclasses

from unitrack.data import Detections, FrameContext, Gate, Tracklets
from unitrack.states.kalman.project import mahalanobis_d2

__all__ = ["MotionGate"]


@dataclasses.dataclass(frozen=True, slots=True)
class MotionGate:
    """
    Mahalanobis chi-squared gate over a Kalman covariance field.

    For each (tracklet, detection) pair, the squared Mahalanobis distance
    ``d2 = (x - z)^T S^{-1} (x - z)`` is computed from the predicted
    tracklet mean ``x``, the projected covariance ``S`` on ``cov_field``,
    and the detection measurement ``z``. Pairs with ``d2 <= max_chi2``
    are admitted; the rest are rejected.

    Parameters
    ----------
    mean_field : str
        Name of the field on :class:`~unitrack.data.Tracklets` and
        :class:`~unitrack.data.Detections` holding the state mean used to
        form the residual.
    cov_field : str
        Name of the field on :class:`~unitrack.data.Tracklets` holding the projected
        measurement covariance ``S``.
    max_chi2 : float
        Chi-squared threshold on the squared Mahalanobis distance. A
        common choice is the 0.95 quantile of the chi-squared
        distribution at the measurement dimensionality (e.g. ``9.4877``
        for 4 degrees of freedom, as used by SORT/DeepSORT).

    """

    mean_field: str
    cov_field: str
    max_chi2: float

    def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
        """Return a PerPair gate keeping pairs with chi2 <= max_chi2."""
        del ctx
        a = getattr(cs, self.mean_field)
        cov = getattr(cs, self.cov_field)
        b = getattr(ds, self.mean_field)
        d2 = mahalanobis_d2(a, cov, b, label="MotionGate")
        return Gate.PerPair(mask=d2 <= self.max_chi2)
