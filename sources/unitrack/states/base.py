"""State, Process, Observation, and Initializer protocols."""

from __future__ import annotations

import dataclasses
import typing

import torch

from unitrack.data import (
    Detections,
    FrameContext,
    MatchOutcome,
    TensorSpec,
    Tracklets,
)

__all__ = ["Initializer", "Observation", "Process", "State"]


@typing.runtime_checkable
class Process(typing.Protocol):
    """
    Predict-step protocol: advance a tracklet field by one time step.

    Implementations are pure functions of ``(cs, ctx)``; they read the
    named field on the tracklet snapshot and return a new snapshot with
    the field updated by ``ctx.delta``.
    """

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """
        Advance the named field by ``ctx.delta``.

        Parameters
        ----------
        cs : Tracklets
            Current tracklet snapshot.
        ctx : FrameContext
            Frame context; ``ctx.delta`` carries the elapsed time.

        Returns
        -------
        Tracklets
            New snapshot with the field advanced.

        """
        ...


@typing.runtime_checkable
class Observation(typing.Protocol):
    """
    Update-step protocol: fuse detection measurements into tracklets.

    Implementations are pure functions of ``(cs, ds, match, ctx)``; they
    consume the matched-pair index from ``match`` and produce a new
    snapshot with the corresponding field updated.
    """

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """
        Fuse matched-detection measurements into tracklets.

        Parameters
        ----------
        cs : Tracklets
            Current tracklet snapshot.
        ds : Detections
            Current-frame detections.
        match : MatchOutcome
            Matched pairs and residuals.
        ctx : FrameContext
            Frame context.

        Returns
        -------
        Tracklets
            New snapshot with the field updated for matched tracklets.

        """
        ...


@typing.runtime_checkable
class Initializer(typing.Protocol):
    """Produce field-shaped initial values for newly-promoted tracklets."""

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """
        Return initial tensor values for new tracklets.

        Parameters
        ----------
        ds : Detections
            Detections promoted to new tracklets this frame.
        ctx : FrameContext
            Frame context.

        Returns
        -------
        torch.Tensor
            Field-shaped initial values, one row per new tracklet.

        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class State:
    """
    A named field on the tracklet snapshot with predict/update/init logic.

    The field name is set by the :class:`~unitrack.tracker.Tracker`'s
    ``states={...}`` dict key and is therefore not stored on the
    :class:`~unitrack.states.State` itself, so the same
    :class:`~unitrack.states.State` instance can be reused under different keys.

    Parameters
    ----------
    schema : ~unitrack.data.TensorSpec
        Per-tracklet tensor shape and dtype for the field.
    process : Process
        Predict-step implementation.
    observation : Observation
        Update-step implementation.
    init : Initializer
        Factory for new-tracklet initial values.

    """

    schema: TensorSpec
    process: Process
    observation: Observation
    init: Initializer
