"""EMA-family :class:`~unitrack.states.Process` and :class:`Observation` primitives."""

from __future__ import annotations

import dataclasses
import math

from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets

__all__ = ["EMADecay", "EMAFuse", "EMATrack", "WeightedFuse"]


@dataclasses.dataclass(frozen=True, slots=True)
class EMADecay:
    """
    Exponential decay of a tracklet field toward ``anchor``.

    Each call multiplies the deviation from ``anchor`` by
    ``exp(-ln 2 * dt / half_life)``, so the field's distance to
    ``anchor`` halves every ``half_life`` time units.

    Parameters
    ----------
    field : str
        Tracklet field to decay.
    half_life : float
        Half-life in seconds. Must be strictly positive.
    anchor : float, optional
        Decay target. Default ``0.0``.

    Raises
    ------
    ValueError
        If ``half_life`` is non-positive or ``anchor`` is non-finite.

    """

    field: str
    half_life: float
    anchor: float = 0.0

    def __post_init__(self) -> None:
        """Validate ``half_life`` and ``anchor``."""
        if self.half_life <= 0:
            msg = f"EMADecay.half_life must be positive; got {self.half_life}"
            raise ValueError(msg)
        if not math.isfinite(self.anchor):
            msg = f"EMADecay.anchor must be finite; got {self.anchor}"
            raise ValueError(msg)

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """
        Decay the field toward :attr:`anchor` by one time step.

        Parameters
        ----------
        cs : Tracklets
            Current tracklet snapshot.
        ctx : FrameContext
            Frame context; ``ctx.delta`` carries the elapsed time.

        Returns
        -------
        Tracklets
            New snapshot with the field decayed.

        Raises
        ------
        ValueError
            If ``ctx.delta`` is negative.

        """
        dt = float(ctx.delta.item())
        if dt < 0:
            msg = (
                f"EMADecay({self.field!r}): ctx.delta must be non-negative; got {dt}. "
                "Time must not run backwards."
            )
            raise ValueError(msg)
        decay = math.exp(-math.log(2.0) * dt / self.half_life)
        v = getattr(cs, self.field)
        return cs.set(self.field, self.anchor + (v - self.anchor) * decay)


@dataclasses.dataclass(frozen=True, slots=True)
class EMATrack:
    """
    Predict-step companion to :class:`EMAFuse`.

    No-op: the EMA blend happens in the observation step. This class
    exists so :class:`EMAFuse` can be wired into a :class:`~unitrack.states.State` whose
    :class:`~unitrack.states.Process` slot expects a non-``None`` predict step.

    Parameters
    ----------
    field : str
        Tracklet field (unused; kept for symmetry with the observation).

    """

    field: str

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Return tracklets unchanged; the EMA blend happens in :class:`EMAFuse`."""
        del ctx
        return cs


@dataclasses.dataclass(frozen=True, slots=True)
class EMAFuse:
    """
    Exponential-moving-average update of a tracklet field.

    For each matched pair the tracklet field is replaced by
    ``rho * field + (1 - rho) * measurement``.

    Parameters
    ----------
    field : str
        Tracklet field to blend.
    rho : float
        Smoothing factor in ``[0, 1]``. Higher values keep more of the
        tracklet history; ``rho = 0`` reduces to :class:`~unitrack.states.Replace`.

    """

    field: str
    rho: float

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Blend matched detections into tracklets via exponential moving average."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        new = getattr(cs, self.field).clone()
        new[cs_idx] = (
            self.rho * new[cs_idx] + (1.0 - self.rho) * getattr(ds, self.field)[ds_idx]
        )
        return cs.set(self.field, new)


@dataclasses.dataclass(frozen=True, slots=True)
class WeightedFuse:
    """
    Score-aware blend that uses a per-detection score as the blend weight.

    Each tracklet field is blended as ``(1 - w) * field + w * detection``,
    where ``w`` is the detection's ``weight_field`` clamped to ``[0, 1]``.

    Parameters
    ----------
    field : str
        Tracklet field to blend.
    weight_field : str
        Detection field holding the per-detection blend weight (e.g. a
        score in ``[0, 1]``).

    """

    field: str
    weight_field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Blend matched detections using per-detection score as blend weight."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        w = getattr(ds, self.weight_field)[ds_idx].clamp(0.0, 1.0)
        new = getattr(cs, self.field).clone()
        new[cs_idx] = (1.0 - w[:, None]) * new[cs_idx] + w[:, None] * getattr(
            ds, self.field
        )[ds_idx]
        return cs.set(self.field, new)
