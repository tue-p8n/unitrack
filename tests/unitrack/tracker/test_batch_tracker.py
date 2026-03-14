# tests/unitrack/tracker/test_batch_tracker.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import BatchTracker, Tracker


def _build():
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


def _ds(kernel):
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_batch_tracker_matches_independent_streams_per_slot():
    tr = _build()
    bt = BatchTracker(tr, batch_size=2)

    ds_per_slot = [
        _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),  # slot 0
        _ds(torch.tensor([[0.0, 1.0]])),  # slot 1
    ]
    ctx_per_slot = [
        FrameContext.make(0, stream_key=0),
        FrameContext.make(0, stream_key=1),
    ]
    results = bt.step(ds_per_slot, ctx_per_slot)

    assert len(results) == 2
    # Each slot got fresh IDs starting at 1.
    assert sorted(results[0].ids.tolist()) == [1, 2]
    assert sorted(results[1].ids.tolist()) == [1]


def test_batch_tracker_isolates_state_across_slots():
    tr = _build()
    bt = BatchTracker(tr, batch_size=2)

    bt.step(
        [_ds(torch.tensor([[1.0, 0.0]])), _ds(torch.tensor([[0.0, 1.0]]))],
        [FrameContext.make(0, stream_key=0), FrameContext.make(0, stream_key=1)],
    )
    # Now feed slot 0 detections that are far from slot 0's tracklet — should
    # spawn a new ID; slot 1 should be unchanged because it sees the same
    # detection it did last frame.
    bt.step(
        [_ds(torch.tensor([[-1.0, 0.0]])), _ds(torch.tensor([[0.0, 1.0]]))],
        [FrameContext.make(1, stream_key=0), FrameContext.make(1, stream_key=1)],
    )
    # Slot 0 has 2 tracklets now (old + new); slot 1 still has 1.
    assert bt.snapshot_of(0).batch_size[0] == 2
    assert bt.snapshot_of(1).batch_size[0] == 1
