"""ClipAssociator abstract base — extension point for clip-global solvers."""

from __future__ import annotations

import abc

from unitrack.data import (
    ClipDetections,
    ClipFrameContext,
    ClipMatchOutcome,
    ClipTracklets,
)

__all__ = ["ClipAssociator"]


class ClipAssociator(abc.ABC):
    """
    Abstract base for clip-global matchers.

    Extension point: no concrete implementation ships in-tree. Users
    provide their own subclass and pass it to :class:`~unitrack.tracker.ClipTracker`.
    """

    @abc.abstractmethod
    def __call__(
        self,
        cs: ClipTracklets,
        ds: ClipDetections,
        ctx: ClipFrameContext,
    ) -> ClipMatchOutcome:
        """
        Match clip tracklets to clip detections.

        Parameters
        ----------
        cs : ClipTracklets
            Tracklet snapshot spanning the clip.
        ds : ClipDetections
            Detection snapshot spanning the clip.
        ctx : ClipFrameContext
            Per-clip frame context.

        Returns
        -------
        ClipMatchOutcome
            Per-frame match outcomes for the clip.

        """
        ...
