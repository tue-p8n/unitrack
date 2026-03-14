# tests/unitrack/tracker/test_differentiable_states.py
"""Regression tests for I6: differentiable=True walks states and lifecycle."""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker, SoftAssignment
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import (
    ConfirmedOnly,
    SoftLifecycle,
    StandardLifecycle,
)
from unitrack.pipeline import Pipe
from unitrack.states import (
    FromDetectionField,
    Identity,
    Replace,
    SoftReplace,
    State,
)
from unitrack.tracker import Tracker


def _build_kernel_state():
    return State(
        schema=TensorSpec(shape=(2,), dtype=torch.float32),
        process=Identity("kernel"),
        observation=Replace("kernel"),
        init=FromDetectionField("kernel"),
    )


def test_differentiable_swaps_replace_to_soft_replace():
    """The default registry maps Replace → SoftReplace inside every State."""
    tr = Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states={"kernel": _build_kernel_state()},
        lifecycle=StandardLifecycle(min_hits=1, max_age=3),
        visibility=ConfirmedOnly(),
        differentiable=True,
    )
    assert isinstance(tr.states["kernel"].observation, SoftReplace)


def test_differentiable_swaps_standard_lifecycle_to_soft_lifecycle():
    """The default registry maps StandardLifecycle → SoftLifecycle."""
    tr = Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states={"kernel": _build_kernel_state()},
        lifecycle=StandardLifecycle(min_hits=1, max_age=3),
        visibility=ConfirmedOnly(),
        differentiable=True,
    )
    assert isinstance(tr.lifecycle, SoftLifecycle)


def test_soft_step_attaches_soft_plan_and_runs_soft_replace():
    """E2E: Associate(SoftAssignment) attaches a plan; SoftReplace consumes it."""
    tr = Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=2.0))),
        states={"kernel": _build_kernel_state()},
        lifecycle=StandardLifecycle(min_hits=1, max_age=3),
        visibility=ConfirmedOnly(),
        differentiable=True,
    )
    # Confirm SoftAssignment is the inner backend.
    assert isinstance(tr.root.assoc.assignment, SoftAssignment)

    # Frame 0: spawn two tracklets.
    snap = tr.empty_snapshot()
    dets0 = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    res = tr.step(snap, dets0, FrameContext.make(0), next_id=1)
    snap = res.snapshot

    # Frame 1: nearly the same — soft replace should blend matched detections in.
    dets1 = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        batch_size=[2],
    )
    res = tr.step(snap, dets1, FrameContext.make(1), next_id=res.next_id)
    # Soft path keeps every tracklet alive (no row drops).
    assert res.snapshot.batch_size[0] >= 2


def test_soft_lifecycle_does_not_drop_rows():
    """SoftLifecycle keeps the snapshot row count stable across frames."""
    from unitrack.data import MatchOutcome, Tracklets
    from unitrack.lifecycle import TrackletStatus

    soft = SoftLifecycle(min_hits=2, max_age=1, grace_period=0)
    cs = Tracklets(
        id=torch.tensor([1], dtype=torch.int64),
        status=torch.tensor([int(TrackletStatus.Tentative)], dtype=torch.int8),
        hits=torch.ones(1, dtype=torch.int32),
        time_since_update=torch.zeros(1, dtype=torch.int32),
        age=torch.tensor([1], dtype=torch.int32),
        frame_started=torch.zeros(1, dtype=torch.int32),
        frame_last_seen=torch.zeros(1, dtype=torch.int32),
        batch_size=[1],
    )
    no_match = MatchOutcome(
        matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
        tracklets_residual_index=torch.arange(1, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    out = soft(cs, no_match, FrameContext.make(1))
    # Hard policy would have removed this Tentative + miss. Soft keeps the row in
    # place (status updates to Removed but the row survives).
    assert out.batch_size[0] == 1
    assert int(out.status[0]) == int(TrackletStatus.Removed)
