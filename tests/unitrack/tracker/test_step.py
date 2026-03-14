# tests/unitrack/tracker/test_step.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import (
    Detections,
    FrameContext,
    TensorSpec,
)
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import StepResult, Tracker


def _make_tracker() -> Tracker:
    return Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states={
            "kernel": State(
                schema=TensorSpec(shape=(2,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


def test_first_frame_creates_two_tracklets_with_fresh_ids():
    tr = _make_tracker()
    snap = tr.empty_snapshot()
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    res = tr.step(snap, ds, FrameContext.make(0, delta=0.0), next_id=1)
    assert isinstance(res, StepResult)
    assert res.snapshot.batch_size[0] == 2
    assert sorted(res.ids.tolist()) == [1, 2]
    assert res.next_id == 3


def test_visibility_remaps_matches_through_lifecycle_filter():
    # Regression for C2: with StandardLifecycle pruning rows, visibility must
    # see ``match.matched_pairs`` indices remapped into the post-lifecycle row
    # space; otherwise ``ConfirmedOnly`` either crashes (out-of-bounds) or
    # returns IDs for the wrong tracklets.
    from unitrack.lifecycle import ConfirmedOnly, StandardLifecycle

    tr = Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states={
            "kernel": State(
                schema=TensorSpec(shape=(2,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=StandardLifecycle(min_hits=1, max_age=0),
        visibility=ConfirmedOnly(),
    )
    snap = tr.empty_snapshot()
    # Frame 0: spawn two Tentative+matched-by-construction tracklets; they
    # promote to Active immediately (min_hits=1).
    ds_t0 = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    res_t0 = tr.step(snap, ds_t0, FrameContext.make(0, delta=1.0), next_id=1)
    # Frame 1: only one detection — the second Active tracklet misses and
    # transitions to Lost. The first stays matched/Active.
    ds_t1 = Detections(
        index=torch.arange(1, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0]]),
        batch_size=[1],
    )
    res_t1 = tr.step(
        res_t0.snapshot,
        ds_t1,
        FrameContext.make(1, delta=1.0),
        next_id=res_t0.next_id,
    )
    # ConfirmedOnly should expose exactly the matched Active tracklet (id=1).
    assert res_t1.ids.tolist() == [1]


def test_second_frame_reuses_ids_for_matched_tracklets():
    tr = _make_tracker()
    snap = tr.empty_snapshot()

    ds_t0 = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    res_t0 = tr.step(snap, ds_t0, FrameContext.make(0), next_id=1)

    ds_t1 = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),  # swapped order
        batch_size=[2],
    )
    res_t1 = tr.step(
        res_t0.snapshot,
        ds_t1,
        FrameContext.make(1, delta=1 / 30),
        next_id=res_t0.next_id,
    )

    # The IDs the caller gets back must reflect the per-detection identity
    # mapping after matching.
    assert res_t1.snapshot.id.tolist() == sorted(res_t0.snapshot.id.tolist())
    assert res_t1.next_id == res_t0.next_id
