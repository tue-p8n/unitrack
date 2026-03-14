"""Constant-velocity centroid Kalman process specializations."""

from __future__ import annotations

import dataclasses

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

__all__ = ["KalmanCentroid2D", "KalmanCentroid3D"]


def _cv_f(spatial_dim: int, dt: float) -> torch.Tensor:
    """Build constant-velocity state-transition matrix."""
    d = 2 * spatial_dim
    f = torch.eye(d)
    for i in range(spatial_dim):
        f[i, spatial_dim + i] = dt
    return f


def _cv_h(spatial_dim: int) -> torch.Tensor:
    """Build measurement matrix that observes position only."""
    d = 2 * spatial_dim
    h = torch.zeros(spatial_dim, d)
    for i in range(spatial_dim):
        h[i, i] = 1.0
    return h


def _cv_q(spatial_dim: int, q: float) -> torch.Tensor:
    """Build isotropic process-noise covariance."""
    return torch.eye(2 * spatial_dim) * q


def _cv_r(spatial_dim: int, r: float) -> torch.Tensor:
    """Build isotropic measurement-noise covariance."""
    return torch.eye(spatial_dim) * r


@dataclasses.dataclass(frozen=True, slots=True)
class KalmanCentroid2D:
    """
    Constant-velocity 2-D centroid Kalman process.

    State is 4-D ``[x, y, vx, vy]``; measurement is 2-D ``[x, y]``. The
    state-transition matrix injects ``dt`` into the position-velocity
    cross-terms, and ``H`` reads the leading two entries (position).
    Process noise is parameterised as ``q * I_4`` and integrated as
    per-unit-time noise (see :class:`KalmanLinear`); measurement noise is
    ``r * I_2``.

    Parameters
    ----------
    field : str
        Field name for the centroid mean. Covariance lives in
        ``f"{field}_cov"``.
    q : float
        Per-unit-time process-noise scale.
    r : float
        Measurement-noise scale.

    """

    field: str = "centroid"
    q: float = 0.01
    r: float = 0.1

    @property
    def cov_field(self) -> str:
        """Return auxiliary covariance field name."""
        return f"{self.field}_cov"

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Advance centroid mean and covariance by one predict step."""
        dt = float(ctx.delta.item())
        proc = KalmanLinear(
            field=self.field,
            F=_cv_f(2, dt),
            H=_cv_h(2),
            Q=_cv_q(2, self.q),
            R=_cv_r(2, self.r),
        )
        return proc(cs, ctx)

    def make_update(self) -> KalmanUpdate:
        """Construct a matching KalmanUpdate for the 2D centroid observation."""
        return KalmanUpdate(
            field=self.field,
            cov_field=self.cov_field,
            H=_cv_h(2),
            R=_cv_r(2, self.r),
        )

    def state_entries(
        self,
        *,
        meas_field: str | None = None,
        init_cov_scale: float = 1.0,
    ) -> dict[str, State]:
        """
        Return ``(mean, cov)`` :class:`~unitrack.states.State` entries for ``Tracker``.

        The mean entry holds a 4-D ``[x, y, vx, vy]`` state seeded from
        ``meas_field``; the cov entry holds a 4-by-4 covariance no-op'd
        through predict/update because the mean entry's
        :class:`KalmanLinear` / :class:`KalmanUpdate` already write the
        covariance as a side effect.

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
            ``self.field`` and ``f"{self.field}_cov"``.

        """
        meas_field = meas_field or self.field
        return _make_kalman_entries(
            field=self.field,
            full_dim=4,
            process=self,
            update=self.make_update(),
            meas_field=meas_field,
            init_cov_scale=init_cov_scale,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class KalmanCentroid3D:
    """
    Constant-velocity 3-D centroid Kalman process.

    State is 6-D ``[x, y, z, vx, vy, vz]``; measurement is 3-D
    ``[x, y, z]``. The state-transition matrix injects ``dt`` into each
    position-velocity cross-term, and ``H`` reads the leading three entries
    (position). Process noise is parameterised as ``q * I_6`` and
    integrated as per-unit-time noise; measurement noise is ``r * I_3``.

    Parameters
    ----------
    field : str
        Field name for the centroid mean. Covariance lives in
        ``f"{field}_cov"``.
    q : float
        Per-unit-time process-noise scale.
    r : float
        Measurement-noise scale.

    """

    field: str = "centroid"
    q: float = 0.01
    r: float = 0.1

    @property
    def cov_field(self) -> str:
        """Return auxiliary covariance field name."""
        return f"{self.field}_cov"

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Advance centroid mean and covariance by one predict step."""
        dt = float(ctx.delta.item())
        proc = KalmanLinear(
            field=self.field,
            F=_cv_f(3, dt),
            H=_cv_h(3),
            Q=_cv_q(3, self.q),
            R=_cv_r(3, self.r),
        )
        return proc(cs, ctx)

    def make_update(self) -> KalmanUpdate:
        """Construct a matching KalmanUpdate for the 3D centroid observation."""
        return KalmanUpdate(
            field=self.field,
            cov_field=self.cov_field,
            H=_cv_h(3),
            R=_cv_r(3, self.r),
        )

    def state_entries(
        self,
        *,
        meas_field: str | None = None,
        init_cov_scale: float = 1.0,
    ) -> dict[str, State]:
        """
        Return ``(mean, cov)`` :class:`~unitrack.states.State` entries for ``Tracker``.

        The mean entry holds a 6-D ``[x, y, z, vx, vy, vz]`` state seeded
        from ``meas_field``; the cov entry holds a 6-by-6 covariance.

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
            ``self.field`` and ``f"{self.field}_cov"``.

        """
        meas_field = meas_field or self.field
        return _make_kalman_entries(
            field=self.field,
            full_dim=6,
            process=self,
            update=self.make_update(),
            meas_field=meas_field,
            init_cov_scale=init_cov_scale,
        )


def _make_kalman_entries(  # noqa: PLR0913
    *,
    field: str,
    full_dim: int,
    process,
    update: KalmanUpdate,
    meas_field: str,
    init_cov_scale: float,
) -> dict[str, State]:
    """Build the (mean, cov) state-entry pair for a Kalman process."""
    return {
        field: State(
            schema=TensorSpec(shape=(full_dim,), dtype=torch.float32),
            process=process,
            observation=update,
            init=PadZerosInitializer(field=meas_field, full_dim=full_dim),
        ),
        f"{field}_cov": State(
            schema=TensorSpec(shape=(full_dim, full_dim), dtype=torch.float32),
            process=NoopProcess(),
            observation=NoopObservation(),
            init=EyeInitializer(dim=full_dim, scale=init_cov_scale),
        ),
    }
