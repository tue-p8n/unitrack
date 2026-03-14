"""Stage protocols and the construction-time type error."""

from __future__ import annotations

import typing

import torch

from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    Gate,
    MatchOutcome,
    Tracklets,
)

__all__ = [
    "Associator",
    "CostProducer",
    "GateProducer",
    "Lifecycle",
    "PipelineTypeError",
    "Stage",
    "Visibility",
]


@typing.runtime_checkable
class GateProducer(typing.Protocol):
    """Leaf stage that emits a :class:`~unitrack.data.Gate`."""

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> Gate:
        """
        Compute the gate.

        Parameters
        ----------
        cs
            Tracklet snapshot.
        ds
            Detection record.
        ctx
            Frame context.

        Returns
        -------
        Gate
            Per-pair, per-tracklet, per-detection, or cost-bias gate.

        """
        ...


@typing.runtime_checkable
class CostProducer(typing.Protocol):
    """Leaf stage that emits a :class:`~unitrack.data.CostExpression`."""

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the cost expression.

        Parameters
        ----------
        cs
            Tracklet snapshot.
        ds
            Detection record.
        ctx
            Frame context.

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix with optional un-applied gates and
            bias.

        """
        ...


@typing.runtime_checkable
class Associator(typing.Protocol):
    """
    Stage that emits a :class:`~unitrack.data.MatchOutcome`.

    ``ctx`` is threaded through every associator call site by the
    Tracker and forwarded by combinators (:class:`Pipe`,
    :class:`Sequential`, :class:`Iterate`, :class:`Filter`,
    :class:`Gated`) so nested cost / gate producers can read it (e.g.
    for time-aware covariance scaling in
    :class:`~unitrack.costs.Mahalanobis` or
    :class:`~unitrack.gates.MotionGate`). Leaf associators that do not
    need frame context may ignore the argument; the parameter exists
    for forward extensibility (annealed thresholds, per-frame
    telemetry) and to keep stage-tree composition uniform.
    """

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
        cost: CostExpression | None = None,
    ) -> MatchOutcome:
        """
        Run the assignment.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context.
        cost
            Optional pre-computed cost expression from an enclosing
            stage. Leaf associators that produce their own cost ignore
            this argument.

        Returns
        -------
        MatchOutcome
            Matched pairs and residual indices into ``cs`` / ``ds``.

        """
        ...


# Union of the three role protocols. Documented as a type alias;
# enforcement is per-protocol isinstance at combinator construction.
Stage = GateProducer | CostProducer | Associator


class Lifecycle(typing.Protocol):
    """
    Status machine over a merged :class:`~unitrack.data.Tracklets` snapshot.

    Implementations must preserve row order: the returned snapshot must
    be a subset of input rows in their original order (a boolean
    keep-mask applied to ``cs``). :meth:`~unitrack.tracker.Tracker.step`
    exploits this to remap matched-pair indices into the surviving row space; reordering
    or merging rows silently breaks visibility remapping.

    Notes
    -----
    Not :func:`typing.runtime_checkable`. An ``isinstance(_, Lifecycle)``
    check on a ``__call__``-only protocol accepts any callable
    (including ``lambda``), so it adds no real validation. Use as a
    static-typing hint only.

    """

    def __call__(
        self,
        cs: Tracklets,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """
        Apply per-row status transitions and return the surviving rows.

        Parameters
        ----------
        cs
            Merged tracklet snapshot (predicted plus newly-spawned rows).
        match
            Pre-lifecycle match outcome.
        ctx
            Frame context.

        Returns
        -------
        Tracklets
            Row-preserving subset of ``cs``.

        """
        ...


class Visibility(typing.Protocol):
    """Reduce a lifecycle output to the user-facing list of visible IDs."""

    def __call__(self, cs: Tracklets, match: MatchOutcome) -> torch.Tensor:
        """
        Return the IDs of tracklets that should appear in the public output.

        Parameters
        ----------
        cs
            Post-lifecycle snapshot (with Removed rows already dropped
            from the visibility view).
        match
            Pre-lifecycle match outcome remapped into the visible row
            space, optionally extended with virtual pairs for
            newly-spawned tracklets.

        Returns
        -------
        torch.Tensor
            ``int64`` tensor of visible tracklet IDs.

        """
        ...


class PipelineTypeError(TypeError):
    """
    Raised at stage-tree construction when the typed-tree contract is violated.

    Attributes
    ----------
    path : list of str
        Dotted location of the offending node within the tree, suitable
        for inclusion in the error message.

    """

    def __init__(self, message: str, *, path: list[str] | None = None):
        self.path = list(path) if path else []
        loc = ".".join(self.path) if self.path else "<root>"
        full = f"{message} (at: {loc})"
        super().__init__(full)
