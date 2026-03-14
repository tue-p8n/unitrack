"""Tracker.step round-trip tests for empty/edge inputs.

Real detectors regularly emit zero detections (occlusions, scene boundaries);
the tracker must advance lifecycle/age, return an empty match, and keep
unmatched tracklets in the snapshot.
"""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle, StandardLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import Tracker


def _make_tracker(lifecycle=None) -> Tracker:
    return Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states={
            "kernel": State(
                schema=TensorSpec(shape=(4,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=lifecycle or NoLifecycle(),
        visibility=IncludeAll(),
    )


def _empty_detections() -> Detections:
    return Detections(
        index=torch.zeros(0, dtype=torch.int64),
        kernel=torch.zeros((0, 4), dtype=torch.float32),
        batch_size=[0],
    )


def _kernel_detections(values: torch.Tensor) -> Detections:
    return Detections(
        index=torch.arange(values.shape[0], dtype=torch.int64),
        kernel=values,
        batch_size=[values.shape[0]],
    )


def test_first_frame_empty_detections_yields_empty_snapshot():
    tr = _make_tracker()
    res = tr.step(tr.empty_snapshot(), _empty_detections(), FrameContext.make(0), 1)
    assert res.snapshot.batch_size[0] == 0
    assert res.ids.numel() == 0
    assert res.match.matched_pairs.shape == (0, 2)
    assert res.next_id == 1


def test_empty_detections_keeps_existing_tracklets():
    tr = _make_tracker()
    kernel = torch.eye(2, 4, dtype=torch.float32)
    res0 = tr.step(
        tr.empty_snapshot(), _kernel_detections(kernel), FrameContext.make(0), 1
    )
    res1 = tr.step(
        res0.snapshot, _empty_detections(), FrameContext.make(1), res0.next_id
    )
    # Tracklets must survive; no new spawns; no matched pairs.
    assert res1.snapshot.batch_size[0] == 2
    assert res1.snapshot.id.tolist() == res0.snapshot.id.tolist()
    assert res1.match.matched_pairs.shape == (0, 2)
    assert res1.next_id == res0.next_id


def test_empty_detections_advances_age_under_standard_lifecycle():
    tr = _make_tracker(StandardLifecycle(min_hits=1, max_age=10))
    kernel = torch.eye(2, 4, dtype=torch.float32)
    res0 = tr.step(
        tr.empty_snapshot(), _kernel_detections(kernel), FrameContext.make(0), 1
    )
    res1 = tr.step(
        res0.snapshot, _empty_detections(), FrameContext.make(1), res0.next_id
    )
    assert res1.snapshot.batch_size[0] == 2
    # All tracklets aged one frame; tsu incremented (no match).
    assert (res1.snapshot.age == res0.snapshot.age + 1).all()
    assert (
        res1.snapshot.time_since_update == res0.snapshot.time_since_update + 1
    ).all()


def test_empty_detections_eventually_removes_tracklets_past_max_age():
    tr = _make_tracker(StandardLifecycle(min_hits=1, max_age=1))
    kernel = torch.eye(2, 4, dtype=torch.float32)
    res = tr.step(
        tr.empty_snapshot(), _kernel_detections(kernel), FrameContext.make(0), 1
    )
    # Two miss frames: tsu 0 → 1 (alive, == max_age) → 2 (> max_age → Lost).
    for f in range(1, 4):
        res = tr.step(
            res.snapshot, _empty_detections(), FrameContext.make(f), res.next_id
        )
    # Lost tracklets remain until max_age + allow_reid; with allow_reid=0 they
    # transition Lost → Removed and get dropped.
    assert res.snapshot.batch_size[0] == 0


def test_first_frame_zero_detections_then_real_detections():
    """Tracker handles being booted with empty detections at frame 0."""
    tr = _make_tracker()
    res0 = tr.step(tr.empty_snapshot(), _empty_detections(), FrameContext.make(0), 1)
    res1 = tr.step(
        res0.snapshot,
        _kernel_detections(torch.eye(2, 4, dtype=torch.float32)),
        FrameContext.make(1),
        res0.next_id,
    )
    assert res1.snapshot.batch_size[0] == 2
    assert res1.snapshot.id.tolist() == [1, 2]
