"""
Ensemble Kalman filter (deterministic ETKF) for high-dimensional states.

A Kalman filter on a D-dimensional embedding needs a D-by-D covariance,
which is expensive and ill-conditioned when D is large (e.g. a 256-d DETR
query). The Ensemble Kalman Filter sidesteps this: it never forms the
covariance explicitly, representing the belief by an ensemble of ``E``
sample states whose spread *implies* the covariance. Cost scales with the
ensemble size, not ``D``-squared, which is why EnKF is the method of choice
for very high-dimensional filtering (its original home is numerical weather
prediction with millions of dimensions).

The update here is the deterministic **Ensemble Transform Kalman Filter**
(ETKF; Bishop 2001, Hunt 2007) with ``H = I`` and ``R = r I`` and no
localisation: the analysis ensemble is a closed-form linear transform of
the forecast ensemble computed in the ``E``-dimensional ensemble space, so
no random perturbed observations are needed and the step is reproducible.
The predict step is multiplicative covariance inflation, the standard EnKF
treatment of process noise for a random-walk state. Only the one-time
ensemble spawn is randomised, from a fixed seed.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from unitrack.data import (
    Detections,
    FrameContext,
    MatchOutcome,
    TensorSpec,
    Tracklets,
)

from ..base import State
from ..identity import FromDetectionField, NoopObservation, NoopProcess

__all__ = [
    "EnsembleInitializer",
    "EnsembleProcess",
    "EnsembleUpdate",
    "enkf_state_entries",
]


def _symmetric_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """Return the symmetric matrix square root of a batch of SPD matrices."""
    vals, vecs = torch.linalg.eigh(mat)
    sqrt_vals = vals.clamp_min(0.0).sqrt()
    return vecs @ torch.diag_embed(sqrt_vals) @ vecs.transpose(-1, -2)


@dataclasses.dataclass(frozen=True, slots=True)
class EnsembleProcess:
    """
    Random-walk predict step: multiplicative covariance inflation.

    Adds process uncertainty by inflating the ensemble's spread about its
    mean by ``sqrt(1 + q * dt)``. This is the standard EnKF treatment of
    additive process noise for a random-walk state and leaves the ensemble
    mean (the matched estimate) unchanged.

    Parameters
    ----------
    field : str
        Primary mean field. The ensemble members live in
        ``f"{field}_ensemble"`` with shape ``(N, E, D)``.
    q : float
        Per-unit-time process-noise (inflation) scale.

    """

    field: str
    q: float = 0.01

    @property
    def ensemble_field(self) -> str:
        """Return the ensemble-members field name."""
        return f"{self.field}_ensemble"

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Inflate the ensemble spread about its per-tracklet mean."""
        dt = float(ctx.delta.item())
        if dt <= 0.0 or self.q <= 0.0:
            return cs
        members = getattr(cs, self.ensemble_field)  # (N, E, D)
        mean = members.mean(dim=1, keepdim=True)  # (N, 1, D)
        factor = math.sqrt(1.0 + self.q * dt)
        inflated = mean + factor * (members - mean)
        return cs.set(self.ensemble_field, inflated)


@dataclasses.dataclass(frozen=True, slots=True)
class EnsembleUpdate:
    """
    Deterministic ETKF measurement update for matched tracklet-detection pairs.

    Works entirely in the ``E``-dimensional ensemble space (``H = I``,
    ``R = r I``): forms the analysis-error covariance
    ``Pa = [(E-1) I + r^{-1} A A^T]^{-1}``, the mean weights
    ``Pa r^{-1} A (z - x̄)``, and a symmetric-square-root transform of the
    perturbations, then maps back to state space. No random perturbed
    observations. Unmatched tracklets keep their predicted ensemble.

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
    def ensemble_field(self) -> str:
        """Return the ensemble-members field name."""
        return f"{self.field}_ensemble"

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Transform matched ensembles by the deterministic ETKF analysis."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        members = getattr(cs, self.ensemble_field).clone()  # (N, E, D)
        mean = getattr(cs, self.field).clone()  # (N, D)

        xf = members[cs_idx]  # (K, E, D)
        e = xf.shape[1]
        xbar = xf.mean(dim=1)  # (K, D)
        anom = xf - xbar.unsqueeze(1)  # (K, E, D) — perturbations (rows = members)
        z = getattr(ds, self.meas_field)[ds_idx]  # (K, D)

        eye_e = torch.eye(e, device=xf.device, dtype=xf.dtype)
        # C = R^{-1} (H A)^T == anom / r, shaped (K, E, D); CA = C A^T (K, E, E).
        ca = (anom @ anom.transpose(-1, -2)) / self.r  # (K, E, E)
        pa = torch.linalg.inv((e - 1) * eye_e + ca)  # (K, E, E)
        # mean weights: w̄ = Pa C (z - x̄)
        c_innov = (anom @ (z - xbar).unsqueeze(-1)).squeeze(-1) / self.r  # (K, E)
        w_bar = (pa @ c_innov.unsqueeze(-1)).squeeze(-1)  # (K, E)
        # perturbation transform W = sqrt((E-1) Pa) + w̄ (added to every column)
        w_pert = _symmetric_sqrt((e - 1) * pa)  # (K, E, E)
        weights = w_pert + w_bar.unsqueeze(-1)  # (K, E, E)
        # X^a = x̄ + W^T A   (weights act across members)
        analysis = xbar.unsqueeze(1) + weights.transpose(-1, -2) @ anom  # (K, E, D)

        members[cs_idx] = analysis
        mean[cs_idx] = analysis.mean(dim=1)
        return cs.set(self.field, mean).set(self.ensemble_field, members)


@dataclasses.dataclass(frozen=True, slots=True)
class EnsembleInitializer:
    """
    :class:`~unitrack.states.Initializer` spawning an ensemble around a measurement.

    Members are ``z + init_std * noise`` with the noise drawn once from a
    fixed-seed generator and mean-centred so the ensemble mean equals ``z``
    exactly. Only this spawn is randomised; predict and update are
    deterministic.

    Parameters
    ----------
    field : str
        Detection field supplying the measurement.
    ensemble_size : int
        Number of members ``E``.
    dim : int
        Embedding dimensionality ``D``.
    init_std : float
        Standard deviation of the initial ensemble spread.
    seed : int
        Seed for the spawn generator.

    """

    field: str
    ensemble_size: int
    dim: int
    init_std: float
    seed: int

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return ``(N, E, D)`` ensemble members centred on the measurement."""
        del ctx
        z = getattr(ds, self.field)  # (N, D)
        n = z.shape[0]
        gen = torch.Generator(device=z.device).manual_seed(self.seed)
        noise = torch.randn(
            n,
            self.ensemble_size,
            self.dim,
            generator=gen,
            device=z.device,
            dtype=z.dtype,
        )
        noise = noise - noise.mean(dim=1, keepdim=True)  # mean-centre -> mean == z
        return z.unsqueeze(1) + self.init_std * noise


def enkf_state_entries(  # noqa: PLR0913
    field: str,
    *,
    dim: int,
    meas_field: str | None = None,
    ensemble_size: int = 32,
    q: float = 0.01,
    r: float = 0.1,
    init_std: float = 0.3,
    seed: int = 0,
) -> dict[str, State]:
    """
    Build the ``(mean, ensemble)`` state entries for an Ensemble Kalman filter.

    The mean entry holds the ensemble mean ``(dim,)`` used for matching; the
    ensemble entry holds the ``(ensemble_size, dim)`` members, no-op'd
    through predict/update because the mean entry's :class:`EnsembleProcess`
    / :class:`EnsembleUpdate` maintain them.

    Parameters
    ----------
    field : str
        Name of the mean field (and prefix for ``f"{field}_ensemble"``).
    dim : int
        Embedding dimensionality.
    meas_field : str, optional
        Detection field supplying the measurement. Defaults to
        :paramref:`field`.
    ensemble_size : int
        Number of ensemble members.
    q : float
        Per-unit-time process-noise (inflation) scale.
    r : float
        Measurement-noise scale.
    init_std : float
        Standard deviation of the initial ensemble spread.
    seed : int
        Seed for the ensemble spawn generator.

    Returns
    -------
    dict
        Two :class:`~unitrack.states.State` entries keyed by ``field`` and
        ``f"{field}_ensemble"``.

    """
    meas_field = meas_field or field
    return {
        field: State(
            schema=TensorSpec(shape=(dim,), dtype=torch.float32),
            process=EnsembleProcess(field=field, q=q),
            observation=EnsembleUpdate(field=field, meas_field=meas_field, r=r),
            init=FromDetectionField(meas_field),
        ),
        f"{field}_ensemble": State(
            schema=TensorSpec(shape=(ensemble_size, dim), dtype=torch.float32),
            process=NoopProcess(),
            observation=NoopObservation(),
            init=EnsembleInitializer(
                field=meas_field,
                ensemble_size=ensemble_size,
                dim=dim,
                init_std=init_std,
                seed=seed,
            ),
        ),
    }
