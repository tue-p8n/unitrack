"""Differentiable companion to :class:`~unitrack.lifecycle.StandardLifecycle`."""

from __future__ import annotations

import dataclasses

from unitrack.data import FrameContext, MatchOutcome, Tracklets

from .policies import _apply_transitions

__all__ = ["SoftLifecycle"]


@dataclasses.dataclass(frozen=True, slots=True)
class SoftLifecycle:
    """
    Shape-stable :class:`~unitrack.lifecycle.StandardLifecycle` for differentiable mode.

    Applies the same status / counter transitions as the hard policy but
    keeps the row count of the snapshot constant: Removed rows stay in
    place rather than being filtered out, so gradients flow through
    every tracklet across frames without index reshuffling. Consumers
    that want only live rows can still filter via the returned status
    field; the soft path itself is shape-stable.

    Status transitions remain discrete — the lifecycle state machine is
    intrinsically non-differentiable. SoftLifecycle's job is purely to
    avoid the autograd-unfriendly row-removal step. End-to-end soft
    learning rides on the upstream :class:`~unitrack.assignment.SoftAssignment` plan and
    soft observations (e.g. :class:`~unitrack.states.SoftReplace`), not on this policy.
    """

    min_hits: int
    max_age: int
    grace_period: int = 0
    allow_reid: int = 0

    def __call__(
        self,
        cs: Tracklets,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Apply lifecycle transitions without dropping rows."""
        if cs.batch_size[0] == 0:
            return cs
        updated, _removed = _apply_transitions(
            cs,
            match,
            ctx,
            min_hits=self.min_hits,
            max_age=self.max_age,
            grace_period=self.grace_period,
            allow_reid=self.allow_reid,
        )
        return updated
