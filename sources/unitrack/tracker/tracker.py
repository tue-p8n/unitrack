"""Pure tracker — step function over :class:`~unitrack.data.Tracklets` snapshots."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import (
    Detections,
    FrameContext,
    MatchOutcome,
    Tracklets,
)
from unitrack.lifecycle import TrackletStatus
from unitrack.pipeline.base import (
    Associator,
    Lifecycle,
    PipelineTypeError,
    Visibility,
)
from unitrack.pipeline.diff import (
    default_soft_registry,
    validate_soft_tree,
    walk_swap,
    walk_swap_states,
)
from unitrack.states import State

__all__ = ["StepResult", "Tracker"]


_RESERVED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "id",
        "status",
        "hits",
        "time_since_update",
        "age",
        "frame_started",
        "frame_last_seen",
    }
)


def _extend_match_with_new_spawns(
    match: MatchOutcome,
    n_predicted: int,
    new_count: int,
    device: torch.device,
) -> MatchOutcome:
    """
    Augment ``match`` with virtual pairs for newly-spawned tracklets.

    A tracklet spawned this frame from an unmatched detection is
    matched by construction: it was created precisely because a
    detection had no pre-existing tracklet to bind to. Visibility
    policies that filter on "matched this frame" need to see these
    implicit pairs or they would exclude first-frame tracklets that the
    lifecycle promoted to Active (e.g. ``StandardLifecycle`` with
    ``min_hits=1``).

    Virtual pair ``k`` is
    ``(n_predicted + k, detections_residual_index[k])``, placing it in
    the same merged-row space as the real matched pairs. The augmented
    match is for visibility consumption only;
    :attr:`StepResult.match` keeps the original real-only pairs.
    """
    if new_count == 0:
        return match
    new_tracklet_idx = torch.arange(
        n_predicted, n_predicted + new_count, dtype=torch.int64, device=device
    )
    new_pairs = torch.stack([new_tracklet_idx, match.detections_residual_index], dim=1)
    extended_pairs = torch.cat([match.matched_pairs, new_pairs], dim=0)
    return MatchOutcome(
        matched_pairs=extended_pairs,
        tracklets_residual_index=match.tracklets_residual_index,
        detections_residual_index=torch.zeros(0, dtype=torch.int64, device=device),
        per_match_cost=torch.cat(
            [
                match.per_match_cost,
                torch.zeros(new_count, dtype=match.per_match_cost.dtype, device=device),
            ]
        ),
        soft_plan=match.soft_plan,
        batch_size=[],  # type: ignore[unknown-argument]
    )


def _drop_removed(lifecycled: Tracklets) -> tuple[Tracklets, torch.Tensor]:
    """
    Project ``lifecycled`` to a visible view of non-Removed rows.

    Returns the filtered snapshot plus an ``int64`` shape
    ``(n_lifecycled,)`` remap from each row of ``lifecycled`` to its
    row in the filtered view (``-1`` for Removed rows that were
    dropped). Under :class:`HardLifecycle` this is a no-op because
    Removed rows are already gone; under :class:`~unitrack.lifecycle.SoftLifecycle` it
    strips the placeholders that the soft path retained for
    shape-stable gradient flow.
    """
    n_life = lifecycled.batch_size[0]
    if n_life == 0:
        empty_remap = torch.zeros(0, dtype=torch.int64, device=lifecycled.id.device)
        return lifecycled, empty_remap
    non_removed = lifecycled.status != int(TrackletStatus.Removed)
    if bool(non_removed.all().item()):
        return lifecycled, torch.arange(
            n_life, dtype=torch.int64, device=lifecycled.id.device
        )
    visible_view = lifecycled[non_removed]  # type: ignore[invalid-assignment]
    remap = torch.full((n_life,), -1, dtype=torch.int64, device=lifecycled.id.device)
    remap[non_removed] = torch.arange(
        int(non_removed.sum().item()),
        dtype=torch.int64,
        device=lifecycled.id.device,
    )
    return visible_view, remap


def _remap_match_for_visibility(
    match: MatchOutcome,
    merged: Tracklets,
    visible_view: Tracklets,
    *,
    lifecycled_to_visible: torch.Tensor,
) -> MatchOutcome:
    """
    Remap ``match.matched_pairs[:, 0]`` from ``merged`` to ``visible_view`` rows.

    Two filters apply in series:

    1. ID membership: a pair whose tracklet was removed by lifecycle
       (Hard) has no surviving identity in ``visible_view`` and is
       dropped.
    2. Status filter: a pair whose tracklet is present in
       ``visible_view`` but marked Removed (Soft) is also dropped via
       ``lifecycled_to_visible``, which carries ``-1`` for Removed
       rows.

    Residual-index fields are preserved as-is; visibility policies do
    not consult them.
    """
    n_merged = merged.batch_size[0]
    if n_merged == 0 or match.matched_pairs.shape[0] == 0:
        return match

    device = merged.id.device
    n_life = lifecycled_to_visible.shape[0]
    # Step 1: which merged rows have any surviving identity in visible_view.
    keep_mask = torch.isin(merged.id, visible_view.id)
    merged_to_life = torch.full((n_merged,), -1, dtype=torch.int64, device=device)
    # Hard path: visible_view rows are a contiguous subset of lifecycled in
    # original order, so merged-row → lifecycled-row is straight via keep_mask.
    # Soft path: same — lifecycled is the same row order as merged for the
    # rows that survived; the lifecycled_to_visible map then drops Removed.
    merged_to_life[keep_mask] = torch.arange(n_life, dtype=torch.int64, device=device)

    new_first_life = merged_to_life[match.matched_pairs[:, 0]]
    valid_life = new_first_life >= 0
    # Step 2: project lifecycled → visible_view (drops Removed rows).
    new_first = torch.where(
        valid_life,
        lifecycled_to_visible[new_first_life.clamp(min=0)],
        torch.full_like(new_first_life, -1),
    )
    valid = new_first >= 0
    new_pairs = torch.stack([new_first[valid], match.matched_pairs[valid, 1]], dim=1)
    return MatchOutcome(
        matched_pairs=new_pairs,
        tracklets_residual_index=match.tracklets_residual_index,
        detections_residual_index=match.detections_residual_index,
        per_match_cost=match.per_match_cost[valid],
        soft_plan=match.soft_plan,
        batch_size=[],  # type: ignore[unknown-argument]
    )


@dataclasses.dataclass(frozen=True, slots=True)
class StepResult:
    """
    One frame's tracker output.

    Attributes
    ----------
    snapshot : ~unitrack.data.Tracklets
        Post-lifecycle snapshot, returned verbatim by the tracker as
        the new "current state".
    match : ~unitrack.data.MatchOutcome
        Raw match outcome produced by the stage tree before lifecycle
        ran. The first column of :attr:`~unitrack.data.MatchOutcome.matched_pairs`
        indexes into the pre-lifecycle merged snapshot (predicted rows
        in ``[0, N_predicted)`` plus newly-spawned rows in
        ``[N_predicted, N_merged)``), not into :attr:`snapshot`. If
        lifecycle drops any row, those indices may no longer correspond
        to rows in :attr:`snapshot` of the same identity. Look tracklets
        up by ``snapshot.id`` for stable identity-based access.
    ids : torch.Tensor
        ``int64`` tensor of visible-tracklet IDs as selected by the
        configured :class:`~unitrack.pipeline.Visibility` policy.
    next_id : int
        First ID available for the next call to :meth:`~unitrack.tracker.Tracker.step`.

    """

    snapshot: Tracklets
    match: MatchOutcome
    ids: torch.Tensor
    next_id: int


class Tracker:
    """
    Stateless tracker driven by a stage tree.

    The tracker is a pure step function over :class:`~unitrack.data.Tracklets`
    snapshots: it owns no per-stream state. Wrap it in
    :class:`~unitrack.tracker.MultiStream` (per-key state) or :class:`BatchTracker`
    (per-slot state) for stateful inference.

    Parameters
    ----------
    root
        Root :class:`~unitrack.pipeline.Associator`.
    states
        Mapping of state name to :class:`~unitrack.states.State`. Each
        contributes one user field to the tracker's
        :class:`~unitrack.data.Tracklets` schema; state names must not collide with
        reserved Tracklets columns.
    lifecycle
        :class:`~unitrack.pipeline.Lifecycle` callable.
    visibility
        :class:`~unitrack.pipeline.Visibility` callable.
    differentiable
        When ``True``, swap hard nodes in ``root`` / ``states`` /
        ``lifecycle`` for their soft replacements before validating the
        tree.

    Attributes
    ----------
    root : ~unitrack.pipeline.Associator
        Validated stage-tree root.
    states : dict
        Validated state map (a fresh dict, not the input).
    lifecycle : ~unitrack.pipeline.Lifecycle
    visibility : ~unitrack.pipeline.Visibility

    Raises
    ------
    PipelineTypeError
        If ``root`` is not an associator, ``lifecycle`` /
        ``visibility`` are not callable, or any state name shadows a
        reserved Tracklets column.

    """

    def __init__(
        self,
        root: Associator,
        states: dict[str, State],
        lifecycle: Lifecycle,
        visibility: Visibility,
        *,
        differentiable: bool = False,
    ) -> None:
        """See class docstring for parameter semantics."""
        if differentiable:
            registry = default_soft_registry()
            root = walk_swap(root, registry)  # type: ignore[assignment]
            states = walk_swap_states(states, registry)
            lifecycle = walk_swap(lifecycle, registry)  # type: ignore[assignment]
            validate_soft_tree(root, states, lifecycle)
        if not isinstance(root, Associator):
            msg = f"Tracker.root must be an Associator; got {type(root).__name__}"
            raise PipelineTypeError(msg, path=["root"])
        # `lifecycle` and `visibility` are Protocols with `__call__` only;
        # a runtime isinstance check would accept any callable. Static
        # typing carries the contract instead. A misuse fails loudly at the
        # call site with a clear traceback.
        if not callable(lifecycle):
            msg = f"Tracker.lifecycle must be callable; got {type(lifecycle).__name__}"
            raise PipelineTypeError(msg, path=["lifecycle"])
        if not callable(visibility):
            msg = (
                f"Tracker.visibility must be callable; got {type(visibility).__name__}"
            )
            raise PipelineTypeError(msg, path=["visibility"])
        shadowed = _RESERVED_FIELD_NAMES & states.keys()
        if shadowed:
            msg = (
                f"Tracker.states: user field name(s) {sorted(shadowed)!r} would "
                "shadow reserved Tracklets columns. Rename the state(s)."
            )
            raise PipelineTypeError(msg, path=["states"])
        self.root = root
        self.states = dict(states)
        self.lifecycle = lifecycle
        self.visibility = visibility

    def empty_snapshot(self, *, device: torch.types.Device | None = None) -> Tracklets:
        """
        Construct a zero-row :class:`~unitrack.data.Tracklets` matching the schema.

        Parameters
        ----------
        device
            Device for the zero-row tensors.

        Returns
        -------
        ~unitrack.data.Tracklets
            A :class:`~unitrack.data.Tracklets` with ``batch_size=[0]`` whose user
            fields are zero-initialised templates drawn from each
            state's :attr:`~unitrack.states.State.schema`.

        """
        user_fields: dict[str, torch.Tensor] = {
            name: state.schema.empty(slots=0, device=device)
            for name, state in self.states.items()
        }
        z = lambda dt: torch.zeros(0, dtype=dt, device=device)  # noqa: E731
        return Tracklets(
            id=z(torch.int64),
            status=z(torch.int8),
            hits=z(torch.int32),
            time_since_update=z(torch.int32),
            age=z(torch.int32),
            frame_started=z(torch.int32),
            frame_last_seen=z(torch.int32),
            **user_fields,  # type: ignore[invalid-argument-type]
            batch_size=[0],
        )

    def predict_only(self, snapshot: Tracklets, ctx: FrameContext) -> Tracklets:
        """
        Run each state's :class:`~unitrack.states.Process` and return predictions.

        Parameters
        ----------
        snapshot
            Current tracklet snapshot.
        ctx
            Frame context.

        Returns
        -------
        ~unitrack.data.Tracklets
            Snapshot with every state's predict step applied in turn.

        """
        out = snapshot
        for state in self.states.values():
            out = state.process(out, ctx)
        return out

    def step(
        self,
        snapshot: Tracklets,
        detections: Detections,
        ctx: FrameContext,
        next_id: int,
    ) -> StepResult:
        """
        Advance the tracker by one frame.

        Parameters
        ----------
        snapshot
            Current :class:`~unitrack.data.Tracklets` snapshot.
        detections
            New :class:`~unitrack.data.Detections` for this frame.
        ctx
            Frame context.
        next_id
            First available tracklet ID; used for any spawns.

        Returns
        -------
        StepResult
            Post-lifecycle snapshot, raw pre-lifecycle match, visible
            IDs, and the next available ID counter.

        """
        predicted = self.predict_only(snapshot, ctx)
        match: MatchOutcome = self.root(predicted, detections, ctx)
        return self._finalize_step(predicted, detections, match, ctx, next_id)

    def _finalize_step(
        self,
        predicted: Tracklets,
        detections: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
        next_id: int,
    ) -> StepResult:
        """
        Run the post-match half of a frame.

        Performs observe, init, merge, lifecycle, and visibility.
        Shared by :meth:`step` and by
        :meth:`BatchTracker._step_batched`, which assembles its own
        ``match`` from a batched LAP solve and delegates the bookkeeping
        here so the two paths cannot drift silently.

        Parameters
        ----------
        predicted
            Snapshot after :meth:`predict_only`.
        detections
            Frame detections.
        match
            Match outcome from the stage tree.
        ctx
            Frame context.
        next_id
            First available ID for spawned tracklets.

        Returns
        -------
        StepResult
            Final frame output.

        """
        # 1. Observe
        updated = predicted
        for state in self.states.values():
            updated = state.observation(updated, detections, match, ctx)

        # 2. Initialize new tracklets for unmatched detections
        new_ds: Detections = detections[  # type: ignore[invalid-assignment]
            match.detections_residual_index
        ]
        new_count = new_ds.batch_size[0]
        device = updated.id.device
        # Single host sync for frame_idx: every Tensor(frame_idx.item()) call
        # would otherwise force a separate device→host copy on a CUDA snapshot.
        frame_idx_int = int(ctx.frame_idx.item())
        new_ids = torch.arange(
            next_id,
            next_id + new_count,
            dtype=torch.int64,
            device=device,
        )
        new_user_fields = {
            name: state.init(new_ds, ctx) for name, state in self.states.items()
        }
        new_tracklets = Tracklets(
            id=new_ids,
            status=torch.full(
                (new_count,),
                int(TrackletStatus.Tentative),
                dtype=torch.int8,
                device=device,
            ),
            hits=torch.ones(new_count, dtype=torch.int32, device=device),
            time_since_update=torch.zeros(new_count, dtype=torch.int32, device=device),
            age=torch.zeros(new_count, dtype=torch.int32, device=device),
            frame_started=torch.full(
                (new_count,),
                frame_idx_int,
                dtype=torch.int32,
                device=device,
            ),
            frame_last_seen=torch.full(
                (new_count,),
                frame_idx_int,
                dtype=torch.int32,
                device=device,
            ),
            **new_user_fields,  # type: ignore[invalid-argument-type]
            batch_size=[new_count],
        )
        next_id_out = next_id + new_count

        # 3. Merge — tensordict overloads torch.cat for TensorDict arguments
        merged = torch.cat(  # type: ignore[no-matching-overload]
            [updated, new_tracklets], dim=0
        )

        # 4. Lifecycle
        lifecycled = self.lifecycle(merged, match, ctx)

        # 5. Visibility — match.matched_pairs[:, 0] indexes into `merged`'s row
        # space, but `lifecycled` may have dropped rows. Remap matched cs-indices
        # into the surviving `lifecycled` row space (drops any pair whose
        # tracklet was removed by lifecycle). IDs are unique on the snapshot,
        # so the per-row keep mask is recovered via ID membership. The match is
        # also extended with virtual pairs for newly-spawned tracklets so that
        # visibility policies see them as "matched this frame".
        #
        # Under SoftLifecycle, `lifecycled` keeps Removed rows in place for
        # shape-stable gradient flow. Visibility, however, must not see those:
        # `IncludeAll` returns every `cs.id`, so a Removed row would leak through.
        # The "visible view" projection drops Removed rows before invoking the
        # policy. `StepResult.snapshot` still receives the full lifecycled
        # snapshot — the projection is for visibility only.
        extended = _extend_match_with_new_spawns(
            match, predicted.batch_size[0], new_count, updated.id.device
        )
        visible_view, visible_remap = _drop_removed(lifecycled)
        remapped_match = _remap_match_for_visibility(
            extended, merged, visible_view, lifecycled_to_visible=visible_remap
        )
        visible_ids = self.visibility(visible_view, remapped_match)

        return StepResult(
            snapshot=lifecycled, match=match, ids=visible_ids, next_id=next_id_out
        )
