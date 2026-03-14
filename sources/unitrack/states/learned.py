"""
Learned propagation hooks — a slot for a MOTR-style recurrent update.

DETR-based trackers (MOTR, TrackFormer, MeMOTR) do not filter the track
embedding with a hand-written motion model; they let a *learned* module
propagate the track query from frame to frame and fuse the new detection.
That is a learned recurrent filter, not a closed-form one, so unitrack
supports it as a pair of hooks that wrap any callable / ``torch.nn.Module``:

- :class:`LearnedProcess` is the predict step — ``query <- module(query, dt)``.
- :class:`LearnedObservation` is the update step —
  ``query <- module(query, measurement)`` for matched pairs.

Because the wrapped module is autograd-native, these are the differentiable
member of the embedding-filter family: a tracking loss can backpropagate
into the propagation/update module (and through it, the backbone).
"""

from __future__ import annotations

import dataclasses
import typing

from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets

__all__ = ["LearnedObservation", "LearnedProcess"]


@dataclasses.dataclass(frozen=True, slots=True)
class LearnedProcess:
    """
    Predict step that applies a learned module to every tracklet's field.

    Parameters
    ----------
    field : str
        Tracklet field to propagate.
    module : callable
        Callable ``(field_tensor: (N, D), dt: float) -> (N, D)`` — e.g. a
        ``torch.nn.Module`` implementing a recurrent query update. Invoked
        once per frame on all live tracklets.

    """

    field: str
    module: typing.Callable

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Apply the learned propagation module to the field."""
        value = getattr(cs, self.field)
        if value.shape[0] == 0:
            return cs
        dt = float(ctx.delta.item())
        return cs.set(self.field, self.module(value, dt))


@dataclasses.dataclass(frozen=True, slots=True)
class LearnedObservation:
    """
    Update step that fuses matched detections with a learned module.

    Parameters
    ----------
    field : str
        Tracklet field to update.
    meas_field : str
        Detection field holding the measurement.
    module : callable
        Callable ``(track: (K, D), measurement: (K, D)) -> (K, D)`` — e.g. a
        ``GRUCell``-style gated update. Applied to matched pairs only.

    """

    field: str
    meas_field: str
    module: typing.Callable

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Apply the learned update module to matched tracklet-detection pairs."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        new_field = getattr(cs, self.field).clone()
        track = new_field[cs_idx]
        measurement = getattr(ds, self.meas_field)[ds_idx]
        new_field[cs_idx] = self.module(track, measurement)
        return cs.set(self.field, new_field)
