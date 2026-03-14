"""
Information filter — the exact dual of the Kalman filter.

The information form carries the *inverse* covariance instead of the
covariance: an information matrix ``Y = P^{-1}`` and information vector
``y = P^{-1} mu``. Its appeal is the update step, which becomes a plain
*addition* — fusing a measurement only adds ``H^T R^{-1} H`` to ``Y`` and
``H^T R^{-1} z`` to ``y`` — so multiple independent cues combine without a
matrix inverse per fusion. The predict step pays for that by being the
awkward one (it needs a Woodbury identity to add process noise). The
filtered mean ``mu = Y^{-1} y`` is kept in the primary field so the
existing cost zoo can match on it.

This implementation uses a random-walk model (``F = H = I``), which is the
appropriate "no motion" assumption for appearance/kernel embeddings. Its
posterior is the exact Gaussian posterior, identical to
:class:`~unitrack.states.KalmanLinear` + :class:`~unitrack.states.KalmanUpdate`.
"""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import (
    Detections,
    FrameContext,
    MatchOutcome,
    TensorSpec,
    Tracklets,
)

from ..base import State
from ..identity import (
    EyeInitializer,
    FromDetectionField,
    NoopObservation,
    NoopProcess,
    ScaledFromDetectionField,
)

__all__ = [
    "InformationProcess",
    "InformationUpdate",
    "information_state_entries",
]


@dataclasses.dataclass(frozen=True, slots=True)
class InformationProcess:
    r"""
    Random-walk predict step in information form.

    Adds isotropic process noise ``Q = q * dt * I`` to the covariance,
    expressed on the information matrix via the Woodbury identity

    .. math::

        Y' = (Y^{-1} + cI)^{-1} = Y - Y (Y + c^{-1} I)^{-1} Y,
        \quad c = q\,dt,

    then rescales the information vector to preserve the mean
    (``F = I`` leaves ``mu`` unchanged, so ``y' = Y' mu``). A non-positive
    ``dt`` is a no-op.

    Parameters
    ----------
    field : str
        Primary mean field. The information matrix lives in
        ``f"{field}_infomat"`` and the information vector in
        ``f"{field}_infovec"``.
    q : float
        Per-unit-time process-noise scale.

    """

    field: str
    q: float = 0.01

    @property
    def infomat_field(self) -> str:
        """Return the information-matrix field name."""
        return f"{self.field}_infomat"

    @property
    def infovec_field(self) -> str:
        """Return the information-vector field name."""
        return f"{self.field}_infovec"

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Inflate the covariance by ``q * dt`` in information form."""
        dt = float(ctx.delta.item())
        if dt <= 0.0 or self.q <= 0.0:
            return cs
        mean = getattr(cs, self.field)
        info_y = getattr(cs, self.infomat_field)
        c = self.q * dt
        eye = torch.eye(info_y.shape[-1], device=info_y.device, dtype=info_y.dtype)
        # Y' = Y - Y (Y + (1/c) I)^{-1} Y
        inner = info_y + eye / c
        new_y = info_y - info_y @ torch.linalg.solve(inner, info_y)
        new_y = 0.5 * (new_y + new_y.transpose(-1, -2))  # symmetrise
        new_vec = (new_y @ mean.unsqueeze(-1)).squeeze(-1)
        return cs.set(self.infomat_field, new_y).set(self.infovec_field, new_vec)


@dataclasses.dataclass(frozen=True, slots=True)
class InformationUpdate:
    """
    Additive measurement update in information form.

    For each matched pair (``H = I``, ``R = r I``) the fusion is a plain
    sum, ``Y' = Y + r^{-1} I`` and ``y' = y + r^{-1} z``, after which the
    filtered mean is recovered as ``mu = Y'^{-1} y'``. Unmatched tracklets
    keep their predicted state.

    Parameters
    ----------
    field : str
        Primary mean field.
    meas_field : str
        Detection field holding the measurement.
    r : float
        Measurement-noise scale.

    """

    field: str
    meas_field: str
    r: float = 0.1

    @property
    def infomat_field(self) -> str:
        """Return the information-matrix field name."""
        return f"{self.field}_infomat"

    @property
    def infovec_field(self) -> str:
        """Return the information-vector field name."""
        return f"{self.field}_infovec"

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Fuse matched detections by adding measurement information."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        mean = getattr(cs, self.field).clone()
        info_y = getattr(cs, self.infomat_field).clone()
        info_v = getattr(cs, self.infovec_field).clone()
        z = getattr(ds, self.meas_field)[ds_idx]

        dim = info_y.shape[-1]
        eye = torch.eye(dim, device=info_y.device, dtype=info_y.dtype)
        y_upd = info_y[cs_idx] + eye / self.r
        v_upd = info_v[cs_idx] + z / self.r
        mu_upd = torch.linalg.solve(y_upd, v_upd.unsqueeze(-1)).squeeze(-1)

        info_y[cs_idx] = y_upd
        info_v[cs_idx] = v_upd
        mean[cs_idx] = mu_upd
        return (
            cs.set(self.field, mean)
            .set(self.infomat_field, info_y)
            .set(self.infovec_field, info_v)
        )


def information_state_entries(  # noqa: PLR0913
    field: str,
    *,
    dim: int,
    meas_field: str | None = None,
    q: float = 0.01,
    r: float = 0.1,
    init_var: float = 1.0,
) -> dict[str, State]:
    """
    Build the ``(mean, infomat, infovec)`` entries for an information filter.

    The mean entry holds the recovered ``(dim,)`` estimate (match it with
    any embedding cost). The information-matrix and -vector entries are
    no-op'd through predict/update because the mean entry's
    :class:`InformationProcess` / :class:`InformationUpdate` maintain them.

    Parameters
    ----------
    field : str
        Name of the mean field (and prefix for the auxiliary fields).
    dim : int
        Embedding dimensionality.
    meas_field : str, optional
        Detection field supplying the measurement. Defaults to
        :paramref:`field`.
    q : float
        Per-unit-time process-noise scale.
    r : float
        Measurement-noise scale.
    init_var : float
        Initial per-dimension variance; the spawned information matrix is
        ``(1 / init_var) I``.

    Returns
    -------
    dict
        Three :class:`~unitrack.states.State` entries keyed by ``field``,
        ``f"{field}_infomat"`` and ``f"{field}_infovec"``.

    """
    meas_field = meas_field or field
    return {
        field: State(
            schema=TensorSpec(shape=(dim,), dtype=torch.float32),
            process=InformationProcess(field=field, q=q),
            observation=InformationUpdate(field=field, meas_field=meas_field, r=r),
            init=FromDetectionField(meas_field),
        ),
        f"{field}_infomat": State(
            schema=TensorSpec(shape=(dim, dim), dtype=torch.float32),
            process=NoopProcess(),
            observation=NoopObservation(),
            init=EyeInitializer(dim=dim, scale=1.0 / init_var),
        ),
        f"{field}_infovec": State(
            schema=TensorSpec(shape=(dim,), dtype=torch.float32),
            process=NoopProcess(),
            observation=NoopObservation(),
            init=ScaledFromDetectionField(meas_field, scale=1.0 / init_var),
        ),
    }
