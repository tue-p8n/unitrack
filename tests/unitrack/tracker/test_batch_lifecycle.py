# tests/unitrack/tracker/test_batch_lifecycle.py
"""Regression tests for I3 / NTH2: BatchTracker conformance + vmap-safe predicate."""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import (
    ConfirmedOnly,
    IncludeAll,
    NoLifecycle,
    StandardLifecycle,
)
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import BatchTracker, Tracker


def _build(*, lifecycle=None):
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
        lifecycle=lifecycle or NoLifecycle(),
        visibility=IncludeAll() if lifecycle is None else ConfirmedOnly(),
    )


def _ds(kernel):
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_is_vmap_safe_true_for_uniform_no_lifecycle_inputs():
    tr = _build()
    bt = BatchTracker(tr, batch_size=2)
    dets = [_ds(torch.tensor([[1.0, 0.0]])), _ds(torch.tensor([[0.0, 1.0]]))]
    assert bt.is_vmap_safe(dets) is True


def test_is_vmap_safe_false_under_standard_lifecycle():
    tr = _build(lifecycle=StandardLifecycle(min_hits=1, max_age=3))
    bt = BatchTracker(tr, batch_size=2)
    dets = [_ds(torch.tensor([[1.0, 0.0]])), _ds(torch.tensor([[0.0, 1.0]]))]
    assert bt.is_vmap_safe(dets) is False


def test_is_vmap_safe_false_for_unequal_detection_counts():
    tr = _build()
    bt = BatchTracker(tr, batch_size=2)
    dets = [
        _ds(torch.tensor([[1.0, 0.0]])),
        _ds(torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
    ]
    assert bt.is_vmap_safe(dets) is False


def test_batch_tracker_matches_sequential_under_standard_lifecycle():
    """NTH2: per-slot BatchTracker output equals per-slot sequential Tracker.step."""
    tr = _build(lifecycle=StandardLifecycle(min_hits=1, max_age=3))
    bt = BatchTracker(tr, batch_size=2)

    # Build the same inputs for both batched and sequential paths.
    frames = [
        (
            _ds(torch.tensor([[1.0, 0.0]])),
            _ds(torch.tensor([[0.0, 1.0]])),
        ),
        (
            _ds(torch.tensor([[1.0, 0.0]])),
            _ds(torch.tensor([[0.0, 1.0]])),
        ),
    ]
    ctxs = [
        (FrameContext.make(0, stream_key=0), FrameContext.make(0, stream_key=1)),
        (FrameContext.make(1, stream_key=0), FrameContext.make(1, stream_key=1)),
    ]

    # Batched run.
    for (ds_a, ds_b), (ctx_a, ctx_b) in zip(frames, ctxs, strict=True):
        bt.step([ds_a, ds_b], [ctx_a, ctx_b])
    batched_ids = [
        sorted(bt.snapshot_of(0).id.tolist()),
        sorted(bt.snapshot_of(1).id.tolist()),
    ]

    # Sequential reference per slot.
    ref_ids = []
    for slot in (0, 1):
        snap = tr.empty_snapshot()
        next_id = 1
        for fi in range(len(frames)):
            res = tr.step(snap, frames[fi][slot], ctxs[fi][slot], next_id)
            snap = res.snapshot
            next_id = res.next_id
        ref_ids.append(sorted(snap.id.tolist()))

    assert batched_ids == ref_ids
