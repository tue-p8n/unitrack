"""
Multi-slot tracker wrapper with an optional batched LAP fast path.

:meth:`BatchTracker.step` runs either a per-slot fallback loop or, when
:meth:`is_vmap_safe` holds and the root is :class:`Pipe(_, Associate)`,
a batched LAP solve via :func:`auto_batch_assignment`. The opt-in
:meth:`BatchTracker.predict_and_cost_vmap` runs the predict and
cost-production phase under :func:`torch.func.vmap`; end-to-end vmap
currently trips a tensordict-internal unbind limitation tracked by
xfail tests in ``tests/unitrack/tracker/test_batch_vmap.py``.
"""

from __future__ import annotations

import dataclasses

import torch
from tensordict import stack as td_stack

from unitrack.assignment import Associate, auto_batch_assignment
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.lifecycle import NoLifecycle
from unitrack.pipeline.combinators import Pipe

from .memory import TrackletMemory
from .tracker import StepResult, Tracker

__all__ = ["BatchTracker"]


@dataclasses.dataclass(slots=True)
class BatchTracker:
    """
    Run multiple streams' :meth:`~unitrack.tracker.Tracker.step` calls together.

    Two execution paths run side by side. The default loop path makes
    per-slot calls to :meth:`~unitrack.tracker.Tracker.step`; it works for any root,
    lifecycle, and shape configuration. The fast path engages when
    :meth:`is_vmap_safe` holds and the tracker root is a single
    :class:`~unitrack.pipeline.Pipe` wrapping an
    :class:`~unitrack.assignment.Associate`: per-slot cost matrices are
    dispatched together to
    :func:`~unitrack.assignment.auto_batch_assignment`. Cost
    computation, observation, and merge still run per slot; only the
    LAP solve — the expensive part for typical ``N`` — is batched. The
    fast path is a strict optimisation and produces identical
    :class:`StepResult` values to the loop path; the conformance test
    ``tests/unitrack/tracker/test_batch_lifecycle.py`` locks this in.

    Attributes
    ----------
    tracker : Tracker
        Underlying pure tracker.
    batch_size : int
        Number of slots; must be positive.

    """

    tracker: Tracker
    batch_size: int
    _memories: list[TrackletMemory] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        """
        Initialise per-slot :class:`TrackletMemory` instances.

        Raises
        ------
        ValueError
            If :attr:`batch_size` is non-positive.

        """
        if self.batch_size <= 0:
            msg = f"BatchTracker.batch_size must be positive; got {self.batch_size}"
            raise ValueError(msg)
        self._memories = [
            TrackletMemory(self.tracker.empty_snapshot())
            for _ in range(self.batch_size)
        ]

    def snapshot_of(self, slot: int) -> Tracklets:
        """Return the current :class:`~unitrack.data.Tracklets` for ``slot``."""
        return self._memories[slot].snapshot

    def reset(self, slot: int | None = None) -> None:
        """
        Reset one or all slots to their empty state.

        Parameters
        ----------
        slot
            Slot index to reset, or ``None`` to reset every slot.

        """
        if slot is None:
            for mem in self._memories:
                mem.reset()
        else:
            self._memories[slot].reset()

    def is_vmap_safe(
        self,
        detections_per_slot: list[Detections],
    ) -> bool:
        """
        Return ``True`` iff this batch can use the batched-solve fast path.

        Parameters
        ----------
        detections_per_slot
            Per-slot detections for this step.

        Returns
        -------
        bool
            ``True`` only when (a) lifecycle is :class:`~unitrack.lifecycle.NoLifecycle`
            (row-removal would diverge per-slot ``N``), (b) detection
            counts match across slots, (c) tracklet counts match across
            snapshots, and (d) the root is shaped as
            ``Pipe(cost, Associate(...))`` so the cost expression can
            be intercepted between cost-production and the LAP solve.

        """
        if not isinstance(self.tracker.lifecycle, NoLifecycle):
            return False
        if not _all_same_count(detections_per_slot):
            return False
        snap_counts = {mem.snapshot.batch_size[0] for mem in self._memories}
        if len(snap_counts) != 1:
            return False
        root = self.tracker.root
        return isinstance(root, Pipe) and isinstance(root.assoc, Associate)

    def _can_dispatch_batched(
        self,
        detections_per_slot: list[Detections],
    ) -> bool:
        """Stricter than :meth:`is_vmap_safe`: also rules out empty ``N`` or ``M``."""
        if not self.is_vmap_safe(detections_per_slot):
            return False
        m = detections_per_slot[0].batch_size[0] if detections_per_slot else 0
        if m == 0:
            return False
        return self._memories[0].snapshot.batch_size[0] > 0

    def step(
        self,
        detections_per_slot: list[Detections],
        ctx_per_slot: list[FrameContext],
    ) -> list[StepResult]:
        """
        Advance every slot by one frame.

        Parameters
        ----------
        detections_per_slot
            Per-slot detections; length must equal :attr:`batch_size`.
        ctx_per_slot
            Per-slot frame contexts; length must equal
            :attr:`batch_size`.

        Returns
        -------
        list of StepResult
            One entry per slot, in slot order. The fast path is
            selected automatically when ``_can_dispatch_batched``
            holds.

        Raises
        ------
        ValueError
            If the input list lengths disagree with :attr:`batch_size`.

        """
        if len(detections_per_slot) != self.batch_size:
            msg = (
                f"got {len(detections_per_slot)} detection lists, "
                f"expected {self.batch_size}"
            )
            raise ValueError(msg)
        if len(ctx_per_slot) != self.batch_size:
            msg = f"got {len(ctx_per_slot)} contexts, expected {self.batch_size}"
            raise ValueError(msg)

        if self._can_dispatch_batched(detections_per_slot):
            return self._step_batched(detections_per_slot, ctx_per_slot)

        results: list[StepResult] = []
        for slot in range(self.batch_size):
            mem = self._memories[slot]
            res = self.tracker.step(
                mem.snapshot,
                detections_per_slot[slot],
                ctx_per_slot[slot],
                mem.next_id,
            )
            mem.load(res.snapshot, res.next_id)
            results.append(res)
        return results

    def _step_batched(
        self,
        detections_per_slot: list[Detections],
        ctx_per_slot: list[FrameContext],
    ) -> list[StepResult]:
        """
        Run the batched-solve fast path.

        Pre-conditions: :meth:`is_vmap_safe` (re-checked defensively so
        future loosenings do not silently unpin
        :class:`~unitrack.lifecycle.NoLifecycle`).
        The fast path only intercepts the LAP solve; everything
        downstream (observe → init → merge → lifecycle → visibility)
        is delegated to :meth:`Tracker._finalize_step`.

        Raises
        ------
        TypeError
            If the runtime root no longer matches the expected
            ``Pipe(_, Associate)`` shape or the lifecycle is not
            :class:`~unitrack.lifecycle.NoLifecycle`.

        """
        root = self.tracker.root  # type: ignore[assignment]
        if not isinstance(root, Pipe):
            msg = f"_step_batched: root must be Pipe; got {type(root).__name__}"
            raise TypeError(msg)
        if not isinstance(root.assoc, Associate):
            msg = (
                "_step_batched: root.assoc must be Associate; "
                f"got {type(root.assoc).__name__}"
            )
            raise TypeError(msg)
        # NoLifecycle precondition: rows-removed lifecycles diverge per-slot N
        # under the batched solve and the visibility remap inside
        # _finalize_step assumes the row-space is unchanged. Duplicating the
        # is_vmap_safe check so future loosenings don't silently un-pin this
        # invariant.
        if not isinstance(self.tracker.lifecycle, NoLifecycle):
            msg = (
                "_step_batched: lifecycle must be NoLifecycle for batched solve; "
                f"got {type(self.tracker.lifecycle).__name__}"
            )
            raise TypeError(msg)
        threshold = float(root.assoc.assignment.threshold)

        # Phase 1 — predict + cost production per slot.
        predicted_per_slot = []
        materialized: list[torch.Tensor] = []
        for slot in range(self.batch_size):
            mem = self._memories[slot]
            predicted = self.tracker.predict_only(mem.snapshot, ctx_per_slot[slot])
            predicted_per_slot.append(predicted)
            expr = root.cost(predicted, detections_per_slot[slot], ctx_per_slot[slot])
            materialized.append(expr.materialize())

        # Phase 2 — batched LAP solve.
        solver_outputs = auto_batch_assignment(materialized)

        # Phase 3 — per-slot match assembly, then delegate to _finalize_step.
        results: list[StepResult] = []
        for slot in range(self.batch_size):
            pairs, cs_unmatched, ds_unmatched = solver_outputs[slot]
            match = _make_match(
                materialized[slot], pairs, cs_unmatched, ds_unmatched, threshold
            )
            mem = self._memories[slot]
            res = self.tracker._finalize_step(
                predicted_per_slot[slot],
                detections_per_slot[slot],
                match,
                ctx_per_slot[slot],
                mem.next_id,
            )
            mem.load(res.snapshot, res.next_id)
            results.append(res)
        return results

    def predict_and_cost_vmap(
        self,
        detections_per_slot: list[Detections],
        ctx_per_slot: list[FrameContext],
    ) -> tuple[list[Tracklets], list[torch.Tensor]]:
        """
        Run the predict and cost-production phase via :func:`torch.func.vmap`.

        Per-slot snapshots, detections, and contexts are stacked along
        a new leading slot dim with :func:`tensordict.stack`; the
        predict-plus-cost work is then traced under ``torch.func.vmap``.

        Parameters
        ----------
        detections_per_slot
            Per-slot detections.
        ctx_per_slot
            Per-slot frame contexts.

        Returns
        -------
        tuple
            ``(predicted_per_slot, materialized)`` where the first
            element is a list of per-slot predicted :class:`~unitrack.data.Tracklets`
            and the second is a list of per-slot materialised ``(N, M)``
            cost tensors.

        Notes
        -----
        Requires the same preconditions as :meth:`is_vmap_safe` plus
        every stage-tree leaf and state ``Process`` must be vmap-clean
        (no ``.item()`` on per-slot scalars, no Python branches on
        tensor values). Compatible: :class:`~unitrack.states.Identity` with vector cost
        producers (:class:`~unitrack.costs.Cosine`,
        :class:`~unitrack.costs.CDist`, :class:`~unitrack.costs.BiSoftmax`,
        :class:`~unitrack.costs.RBF`). Incompatible:
        :class:`~unitrack.states.KalmanBBox` and the
        :class:`~unitrack.states.KalmanCentroid2D` /
        :class:`~unitrack.states.KalmanCentroid3D`
        variants — they call ``ctx.delta.item()`` when
        building the transition matrix.

        Raises
        ------
        TypeError
            If the underlying root is not ``Pipe(_, Associate)`` or if
            vmap cannot trace the stage tree.

        """
        root = self.tracker.root
        if not isinstance(root, Pipe):
            msg = "predict_and_cost_vmap requires a Pipe-rooted tracker"
            raise TypeError(msg)
        snapshots_stacked = td_stack([mem.snapshot for mem in self._memories], dim=0)
        detections_stacked = td_stack(detections_per_slot, dim=0)
        ctx_stacked = td_stack(ctx_per_slot, dim=0)
        tracker = self.tracker

        def _per_slot(snap: Tracklets, det: Detections, c: FrameContext):
            predicted = tracker.predict_only(snap, c)
            expr = root.cost(predicted, det, c)
            return predicted, expr.materialize()

        predicted_stacked, materialized_stacked = torch.func.vmap(_per_slot)(
            snapshots_stacked, detections_stacked, ctx_stacked
        )
        # torch.func.vmap drops the Tracklets subclass identity on its output
        # (returns a plain TensorDict). Re-wrap each unbound slice as a
        # Tracklets so downstream accessors (cs.kernel etc.) work via the
        # Tracklets __getattr__ fallback.
        predicted_per_slot = [
            _retype_as_tracklets(td) for td in predicted_stacked.unbind(0)
        ]
        materialized = [materialized_stacked[s] for s in range(self.batch_size)]
        return predicted_per_slot, materialized


def _retype_as_tracklets(td) -> Tracklets:
    """
    Re-wrap a plain :class:`TensorDict` as :class:`~unitrack.data.Tracklets`.

    :func:`torch.func.vmap` produces plain :class:`TensorDict` outputs
    even when the input was a :class:`~unitrack.data.Tracklets`-typed stacked view.
    Downstream ``state.observation`` / ``state.init`` accessors
    (``cs.kernel`` etc.) need the :class:`~unitrack.data.Tracklets` subclass to
    dispatch the ``__getattr__`` fallback to the key lookup.
    Reconstructing via the constructor revalidates the reserved-field
    schema, giving a free correctness check on the vmap output.
    """
    if isinstance(td, Tracklets):
        return td
    fields = {k: td[k] for k in td}
    return Tracklets(**fields, batch_size=td.batch_size)


def _make_match(
    materialized: torch.Tensor,
    pairs: torch.Tensor,
    cs_unmatched: torch.Tensor,
    ds_unmatched: torch.Tensor,
    threshold: float,
) -> MatchOutcome:
    """Build a :class:`~unitrack.data.MatchOutcome` from a LAP solver's output."""
    if pairs.shape[0] == 0:
        return MatchOutcome(
            matched_pairs=pairs,
            tracklets_residual_index=cs_unmatched,
            detections_residual_index=ds_unmatched,
            per_match_cost=torch.zeros(
                0, dtype=materialized.dtype, device=materialized.device
            ),
            soft_plan=None,
            batch_size=[],  # type: ignore[unknown-argument]
        )
    pair_costs = materialized[pairs[:, 0], pairs[:, 1]]
    valid = pair_costs <= threshold
    invalid_pairs = pairs[~valid]
    pairs = pairs[valid]
    per_cost = pair_costs[valid]
    cs_unmatched = torch.cat([cs_unmatched, invalid_pairs[:, 0]]).sort().values
    ds_unmatched = torch.cat([ds_unmatched, invalid_pairs[:, 1]]).sort().values
    return MatchOutcome(
        matched_pairs=pairs,
        tracklets_residual_index=cs_unmatched,
        detections_residual_index=ds_unmatched,
        per_match_cost=per_cost,
        soft_plan=None,
        batch_size=[],  # type: ignore[unknown-argument]
    )


def _all_same_count(items: list) -> bool:
    """Return True iff every item in ``items`` has the same ``batch_size[0]``."""
    if not items:
        return True
    first = items[0].batch_size[0]
    return all(it.batch_size[0] == first for it in items)
