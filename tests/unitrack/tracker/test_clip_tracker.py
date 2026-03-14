# tests/unitrack/tracker/test_clip_tracker.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import (
    ClipDetections,
    ClipFrameContext,
    Detections,
    TensorSpec,
)
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import ClipTracker, Tracker


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


def test_per_frame_iterates_step_K_times_and_returns_aligned_clip_tracklets():  # noqa: N802
    tr = _build()
    cliptr = ClipTracker(tr, mode="per_frame")
    snap = tr.empty_snapshot()
    K = 3  # noqa: N806
    frames = []
    for _k in range(K):
        kernel = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        frames.append(
            Detections(
                index=torch.arange(2, dtype=torch.int64), kernel=kernel, batch_size=[2]
            )
        )
    clip = ClipDetections(frames=frames)
    ctx = ClipFrameContext.make(start_frame=0, K=K, fps=15.0)
    _snap_out, clip_track, _clip_match = cliptr.process_clip(snap, clip, ctx)
    assert clip_track.K == K
    # Every frame's snapshot should have >= 2 tracklets.
    for k in range(K):
        assert clip_track.frames[k].batch_size[0] >= 2


def test_reset_per_clip_starts_fresh_each_clip():
    tr = _build()
    cliptr = ClipTracker(tr, mode="per_frame", reset_per_clip=True)
    K = 2  # noqa: N806
    frames = [
        Detections(
            index=torch.arange(1, dtype=torch.int64),
            kernel=torch.tensor([[1.0, 0.0]]),
            batch_size=[1],
        )
        for _ in range(K)
    ]
    clip = ClipDetections(frames=frames)
    ctx = ClipFrameContext.make(start_frame=0, K=K, fps=15.0)
    snap_out, _, _ = cliptr.process_clip(tr.empty_snapshot(), clip, ctx)
    # next call with reset_per_clip starts with fresh IDs
    snap_out2, _, _ = cliptr.process_clip(snap_out, clip, ctx)
    assert snap_out2.batch_size[0] >= 1
