r"""
von Mises-Fisher directional filter for unit-norm embedding states.

A recursive Bayesian filter for appearance/kernel embeddings that live on
the unit sphere (cosine geometry). The belief over a tracklet's embedding is
a von Mises-Fisher distribution with mean direction ``mu`` (a unit vector)
and concentration ``kappa >= 0`` (larger = more certain). The vMF mean
direction has a conjugate vMF prior, so combining the prior with a new
observation is exact: the posterior parameter is the *resultant* of the two
concentration-weighted directions,

.. math::

    R = \kappa\,\mu + \kappa_{obs}\,\hat z,\quad
    \mu' = R / \lVert R \rVert,\quad \kappa' = \lVert R \rVert,

which is the directional analogue of a Kalman update. The predict step has
no motion model for appearance, so it only *decays* the concentration:
confidence in a stale embedding fades with time.
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

from .base import State
from .identity import (
    ConstantInitializer,
    NoopObservation,
    NoopProcess,
    NormalizedFromDetectionField,
)

__all__ = [
    "VonMisesFisherDecay",
    "VonMisesFisherUpdate",
    "vmf_state_entries",
]


@dataclasses.dataclass(frozen=True, slots=True)
class VonMisesFisherDecay:
    """
    Predict step for a von Mises-Fisher embedding state.

    The mean direction is left unchanged (no appearance motion model); the
    concentration decays multiplicatively toward :attr:`kappa_min` with a
    time constant :attr:`tau`, so a tracklet that has not been observed
    recently becomes less certain of its appearance.

    Parameters
    ----------
    field : str
        Tracklet field holding the mean direction. The concentration lives
        in ``f"{field}_kappa"``.
    tau : float
        Decay time constant. Each call multiplies ``kappa`` by
        ``exp(-dt / tau)``. Must be strictly positive.
    kappa_min : float
        Floor on the concentration after decay.

    Raises
    ------
    ValueError
        If ``tau`` is non-positive.

    """

    field: str
    tau: float = 10.0
    kappa_min: float = 1.0

    def __post_init__(self) -> None:
        """Reject a non-positive time constant."""
        if self.tau <= 0:
            msg = f"VonMisesFisherDecay.tau must be positive; got {self.tau}"
            raise ValueError(msg)

    @property
    def kappa_field(self) -> str:
        """Return the auxiliary concentration field name."""
        return f"{self.field}_kappa"

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Decay the concentration by one time step; leave the direction."""
        dt = float(ctx.delta.item())
        kappa = getattr(cs, self.kappa_field)
        decayed = torch.clamp(kappa * math.exp(-dt / self.tau), min=self.kappa_min)
        return cs.set(self.kappa_field, decayed)


@dataclasses.dataclass(frozen=True, slots=True)
class VonMisesFisherUpdate:
    """
    Conjugate von Mises-Fisher update for matched tracklet-detection pairs.

    For each matched pair the posterior is the resultant of the prior
    direction (weighted by its concentration) and the L2-normalised
    detection embedding (weighted by :attr:`kappa_obs`). The new direction
    is the normalised resultant and the new concentration is its length, so
    confident agreement sharpens the belief while disagreement broadens it.
    Unmatched tracklets keep their predicted state.

    Parameters
    ----------
    field : str
        Tracklet field holding the mean direction.
    meas_field : str
        Detection field holding the measured embedding.
    kappa_obs : float
        Observation concentration (how much one detection is trusted).

    """

    field: str
    meas_field: str
    kappa_obs: float = 20.0

    @property
    def kappa_field(self) -> str:
        """Return the auxiliary concentration field name."""
        return f"{self.field}_kappa"

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Fuse matched detections via the conjugate vMF resultant update."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        mu = getattr(cs, self.field).clone()
        kappa = getattr(cs, self.kappa_field).clone()
        z = torch.nn.functional.normalize(getattr(ds, self.meas_field)[ds_idx], dim=-1)
        resultant = kappa[cs_idx, None] * mu[cs_idx] + self.kappa_obs * z
        new_kappa = resultant.norm(dim=-1)
        new_mu = resultant / new_kappa.clamp_min(1e-12)[:, None]
        mu[cs_idx] = new_mu
        kappa[cs_idx] = new_kappa
        return cs.set(self.field, mu).set(self.kappa_field, kappa)


def vmf_state_entries(  # noqa: PLR0913
    field: str,
    *,
    dim: int,
    meas_field: str | None = None,
    init_kappa: float = 20.0,
    kappa_obs: float = 20.0,
    tau: float = 10.0,
    kappa_min: float = 1.0,
) -> dict[str, State]:
    """
    Build the ``(direction, concentration)`` state entries for a vMF filter.

    The direction entry holds a unit ``(dim,)`` mean — match it with the
    existing :class:`~unitrack.costs.Cosine` cost. The concentration entry
    holds a scalar, no-op'd through predict/update because the direction
    entry's :class:`VonMisesFisherDecay` / :class:`VonMisesFisherUpdate`
    write it as a side effect.

    Parameters
    ----------
    field : str
        Name of the direction field (and prefix for ``f"{field}_kappa"``).
    dim : int
        Embedding dimensionality.
    meas_field : str, optional
        Detection field supplying the measured embedding. Defaults to
        :paramref:`field`.
    init_kappa : float
        Concentration assigned to a freshly spawned tracklet.
    kappa_obs : float
        Per-observation concentration used by the update.
    tau : float
        Concentration decay time constant used by the predict step.
    kappa_min : float
        Floor on the concentration after decay.

    Returns
    -------
    dict
        Two :class:`~unitrack.states.State` entries keyed by ``field`` and
        ``f"{field}_kappa"``.

    """
    meas_field = meas_field or field
    return {
        field: State(
            schema=TensorSpec(shape=(dim,), dtype=torch.float32),
            process=VonMisesFisherDecay(field=field, tau=tau, kappa_min=kappa_min),
            observation=VonMisesFisherUpdate(
                field=field, meas_field=meas_field, kappa_obs=kappa_obs
            ),
            init=NormalizedFromDetectionField(meas_field),
        ),
        f"{field}_kappa": State(
            schema=TensorSpec(shape=(), dtype=torch.float32),
            process=NoopProcess(),
            observation=NoopObservation(),
            init=ConstantInitializer(
                TensorSpec(shape=(), dtype=torch.float32), init_kappa
            ),
        ),
    }
