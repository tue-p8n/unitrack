"""KalmanLinear backbone — generic linear-Gaussian predict step."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import FrameContext, Tracklets

__all__ = ["KalmanLinear"]


@dataclasses.dataclass(frozen=True, slots=True)
class KalmanLinear:
    r"""
    Linear-Gaussian Kalman predict step.

    Reads ``cs.{field}`` (mean, shape ``(N, D)``) and ``cs.{field}_cov``
    (covariance, shape ``(N, D, D)``) and writes both back as

    .. math::

        x' = F x, \qquad P' = F P F^T + Q \cdot dt.

    Process-noise scaling
    ---------------------
    ``Q`` is interpreted as a per-unit-time process-noise covariance.
    Each call accumulates ``Q * ctx.delta`` into the covariance so
    variable-rate inputs inject the right amount of uncertainty. This is
    the simplest dt-aware convention that keeps ``Q``'s units physically
    meaningful: integrating Brownian process noise of intensity ``Q`` over
    ``dt`` accumulates variance ``Q * dt``. For a strict
    white-noise-velocity CV discretisation (block ``Q_d`` with
    ``dt^3/3``, ``dt^2/2``, ``dt`` entries) a caller can wrap this class
    and pre-shape ``Q`` per call; the default linear scaling matches what
    SORT, DeepSORT, and most tracking-by-detection pipelines use in
    practice. Pass ``dt_scale_q=False`` to skip the rescaling when the
    caller pre-bakes ``Q`` per frame.

    Parameters
    ----------
    field : str
        Field name for the mean. Covariance lives in ``f"{field}_cov"``.
    F : torch.Tensor
        ``(D, D)`` state-transition matrix.
    H : torch.Tensor
        ``(M, D)`` measurement matrix. Stored here for the paired
        :class:`KalmanUpdate` factory; not used by the predict step
        itself.
    Q : torch.Tensor
        ``(D, D)`` process-noise covariance (per unit time when
        ``dt_scale_q=True``).
    R : torch.Tensor
        ``(M, M)`` measurement-noise covariance. Stored for the paired
        update; not used by the predict step itself.
    dt_scale_q : bool, optional
        Multiply ``Q`` by ``ctx.delta`` before adding to the predicted
        covariance. Default ``True``.

    Raises
    ------
    ValueError
        If ``F``, ``H``, ``Q``, and ``R`` do not all share the same dtype
        (would force silent per-call casting).

    """

    field: str
    F: torch.Tensor  # (D, D)
    H: torch.Tensor  # (M, D)  — measurement matrix (used by Update)
    Q: torch.Tensor  # (D, D)  — process noise (per unit time when dt_scale_q=True)
    R: torch.Tensor  # (M, M)  — measurement noise (used by Update)
    dt_scale_q: bool = True

    def __post_init__(self) -> None:
        """Reject mismatched F/H/Q/R dtypes that would silently down-cast."""
        dtypes = {self.F.dtype, self.H.dtype, self.Q.dtype, self.R.dtype}
        if len(dtypes) != 1:
            msg = (
                f"KalmanLinear: F/H/Q/R must share a dtype to avoid silent "
                f"per-call casting; got {dtypes!r}"
            )
            raise ValueError(msg)

    @property
    def cov_field(self) -> str:
        """Return auxiliary covariance field name."""
        return f"{self.field}_cov"

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Advance mean and covariance by one predict step."""
        mean = getattr(cs, self.field)
        cov = getattr(cs, self.cov_field)
        device = mean.device
        f = self.F.to(device=device, dtype=mean.dtype)
        q = self.Q.to(device=device, dtype=mean.dtype)
        if self.dt_scale_q:
            # Cast delta to mean dtype/device; keep as a 0-d tensor so the
            # multiplication broadcasts and stays on-device.
            dt = ctx.delta.to(device=device, dtype=mean.dtype)
            q = q * dt
        new_mean = mean @ f.T
        new_cov = f @ cov @ f.T + q
        return cs.set(self.field, new_mean).set(self.cov_field, new_cov)
