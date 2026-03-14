"""
Bounding-box Kalman processes — SORT (7-D) and DeepSORT (8-D) variants.

Two models ship in-tree:

- ``"sort"`` (default, 7-D state): ``[x, y, a, h, vx, vy, va]``. Aspect ratio
  ``a`` carries a velocity ``va``; height ``h`` is observed but has no
  velocity (matches the original SORT formulation, where height changes
  are absorbed by the measurement-update step rather than predicted).
- ``"deepsort"`` (8-D state): ``[x, y, a, h, vx, vy, va, vh]``. Adds a
  height-velocity ``vh`` to match the DeepSORT / canonical-Kalman-bbox
  convention used by most modern tracking-by-detection pipelines.

Both models observe the same 4-D measurement ``[x, y, a, h]``.
"""

from __future__ import annotations

import dataclasses
import typing

import torch

from unitrack.data import FrameContext, TensorSpec, Tracklets

from ..base import State
from ..identity import (
    EyeInitializer,
    NoopObservation,
    NoopProcess,
    PadZerosInitializer,
)
from .base import KalmanLinear
from .update import KalmanUpdate

__all__ = ["KalmanBBox"]

BBoxModel = typing.Literal["sort", "deepsort"]


def _sort_f(dt: float) -> torch.Tensor:
    """
    Build SORT 7-dim state-transition matrix.

    State: ``[x, y, a, h, vx, vy, va]``.
    """
    f = torch.eye(7)
    f[0, 4] = dt  # x  <- x  + vx * dt
    f[1, 5] = dt  # y  <- y  + vy * dt
    f[2, 6] = dt  # a  <- a  + va * dt  (a = aspect ratio)
    return f


def _sort_h() -> torch.Tensor:
    """Build SORT measurement matrix (observe x, y, a, h from 7-D state)."""
    h = torch.zeros(4, 7)
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    h[2, 2] = 1.0
    h[3, 3] = 1.0
    return h


def _deepsort_f(dt: float) -> torch.Tensor:
    """
    Build DeepSORT 8-dim state-transition matrix.

    State: ``[x, y, a, h, vx, vy, va, vh]``. Adds height velocity to SORT's
    7-D model, matching the canonical Kalman-bbox formulation used by
    DeepSORT and most modern tracking-by-detection pipelines.
    """
    f = torch.eye(8)
    f[0, 4] = dt  # x <- x + vx*dt
    f[1, 5] = dt  # y <- y + vy*dt
    f[2, 6] = dt  # a <- a + va*dt
    f[3, 7] = dt  # h <- h + vh*dt
    return f


def _deepsort_h() -> torch.Tensor:
    """Build DeepSORT measurement matrix (observe x, y, a, h from 8-D state)."""
    h = torch.zeros(4, 8)
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    h[2, 2] = 1.0
    h[3, 3] = 1.0
    return h


def _model_dim(model: BBoxModel) -> int:
    return 8 if model == "deepsort" else 7


def _model_f(model: BBoxModel, dt: float) -> torch.Tensor:
    return _deepsort_f(dt) if model == "deepsort" else _sort_f(dt)


def _model_h(model: BBoxModel) -> torch.Tensor:
    return _deepsort_h() if model == "deepsort" else _sort_h()


@dataclasses.dataclass(frozen=True, slots=True)
class KalmanBBox:
    """
    Constant-velocity bounding-box Kalman process.

    Builds the state-transition matrix ``F``, measurement matrix ``H``, and
    isotropic noise covariances ``Q``, ``R`` for one of two bbox models:

    - ``"sort"`` — 7-D state ``[x, y, a, h, vx, vy, va]``, 4-D measurement
      ``[x, y, a, h]``. Height carries no velocity, matching the original
      SORT formulation.
    - ``"deepsort"`` — 8-D state ``[x, y, a, h, vx, vy, va, vh]`` with an
      added height velocity, matching the canonical Kalman-bbox convention.

    ``Q`` is parameterised as ``q * I_D`` and treated as per-unit-time
    process noise: each call accumulates ``q * dt`` along the diagonal of
    the propagated covariance (see :class:`KalmanLinear` for the exact
    integration rule). ``R`` is parameterised as ``r * I_4``.

    Parameters
    ----------
    field : str
        Field name for the bbox mean. The matching covariance lives in
        ``f"{field}_cov"``.
    q : float
        Per-unit-time process-noise scale.
    r : float
        Measurement-noise scale.
    model : {"sort", "deepsort"}
        Bbox motion model. Default ``"sort"``.

    Raises
    ------
    ValueError
        If ``model`` is not one of ``"sort"`` or ``"deepsort"``.

    """

    field: str = "bbox"
    q: float = 0.01
    r: float = 0.1
    model: BBoxModel = "sort"

    def __post_init__(self) -> None:
        """Reject unknown bbox models early with a clear message."""
        if self.model not in ("sort", "deepsort"):
            msg = f"KalmanBBox.model must be 'sort' or 'deepsort'; got {self.model!r}"
            raise ValueError(msg)

    @property
    def cov_field(self) -> str:
        """Return auxiliary covariance field name."""
        return f"{self.field}_cov"

    @property
    def state_dim(self) -> int:
        """Return the state dimensionality (7 for SORT, 8 for DeepSORT)."""
        return _model_dim(self.model)

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Advance bbox mean and covariance by one predict step."""
        dt = float(ctx.delta.item())
        d = self.state_dim
        proc = KalmanLinear(
            field=self.field,
            F=_model_f(self.model, dt),
            H=_model_h(self.model),
            Q=torch.eye(d) * self.q,
            R=torch.eye(4) * self.r,
        )
        return proc(cs, ctx)

    def make_update(self) -> KalmanUpdate:
        """Construct a matching KalmanUpdate for the bbox observation."""
        return KalmanUpdate(
            field=self.field,
            cov_field=self.cov_field,
            H=_model_h(self.model),
            R=torch.eye(4) * self.r,
        )

    def state_entries(
        self,
        *,
        meas_field: str | None = None,
        init_cov_scale: float = 1.0,
    ) -> dict[str, State]:
        """
        Return ``(mean, cov)`` :class:`~unitrack.states.State` entries.

        Mean is a 7-D ``[x, y, a, h, vx, vy, va]`` (SORT) or 8-D
        ``[x, y, a, h, vx, vy, va, vh]`` (DeepSORT) state seeded from the
        detection field ``meas_field``; the detection must already be in
        ``[x, y, a, h]`` order — convert from xyxy upstream.

        Parameters
        ----------
        meas_field : str, optional
            Detection field that supplies the initial measurement. Defaults
            to :attr:`field`.
        init_cov_scale : float, optional
            Scale applied to the identity matrix used to initialise the
            covariance. Default ``1.0``.

        Returns
        -------
        dict
            Two :class:`~unitrack.states.State` entries keyed by
            ``self.field`` (mean) and ``f"{self.field}_cov"`` (covariance).

        """
        meas_field = meas_field or self.field
        d = self.state_dim
        return {
            self.field: State(
                schema=TensorSpec(shape=(d,), dtype=torch.float32),
                process=self,
                observation=self.make_update(),
                init=PadZerosInitializer(field=meas_field, full_dim=d),
            ),
            f"{self.field}_cov": State(
                schema=TensorSpec(shape=(d, d), dtype=torch.float32),
                process=NoopProcess(),
                observation=NoopObservation(),
                init=EyeInitializer(dim=d, scale=init_cov_scale),
            ),
        }
