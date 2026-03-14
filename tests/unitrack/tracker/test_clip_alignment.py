# tests/unitrack/tracker/test_clip_alignment.py
"""Regression tests for I4: ClipTracker emits row-aligned ClipTracklets."""

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
from unitrack.lifecycle import (
    ConfirmedOnly,
    IncludeAll,
    NoLifecycle,
    StandardLifecycle,
)
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import ClipTracker, Tracker


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
        visibility=ConfirmedOnly() if lifecycle is not None else IncludeAll(),
    )


def _ds(kernel):
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_clip_tracker_frames_share_row_ordering_by_id():
    """Aligned clip: row n at frame k has the same identity as row n at frame k+1."""
    tr = _build()
    cliptr = ClipTracker(tr, mode="per_frame")

    # Frame 0: one detection.  Frame 1: two detections (a new identity).
    frames = [
        _ds(torch.tensor([[1.0, 0.0]])),
        _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
    ]
    clip = ClipDetections(frames=frames)
    ctx = ClipFrameContext.make(start_frame=0, K=2, fps=15.0)
    _snap_out, clip_t, _ = cliptr.process_clip(tr.empty_snapshot(), clip, ctx)

    # Both frames must have the same number of rows.
    n0, n1 = clip_t.frames[0].batch_size[0], clip_t.frames[1].batch_size[0]
    assert n0 == n1
    # Row-by-row identity alignment.
    assert torch.equal(clip_t.frames[0].id, clip_t.frames[1].id)


def test_clip_tracker_inserts_removed_placeholders_for_absent_ids():
    """Lifecycle-dropped tracklets become Removed placeholders, not missing rows."""
    from unitrack.lifecycle import TrackletStatus

    # StandardLifecycle with max_age=0 + no re-id removes a Lost tracklet
    # after one more frame, so over 3 frames the [0,1] tracklet is filtered
    # out of the raw snapshot by frame 2.
    tr = _build(lifecycle=StandardLifecycle(min_hits=1, max_age=0, allow_reid=0))
    cliptr = ClipTracker(tr, mode="per_frame")

    frames = [
        _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        _ds(torch.tensor([[1.0, 0.0]])),
        _ds(torch.tensor([[1.0, 0.0]])),
    ]
    clip = ClipDetections(frames=frames)
    ctx = ClipFrameContext.make(start_frame=0, K=3, fps=15.0)
    _, clip_t, _ = cliptr.process_clip(tr.empty_snapshot(), clip, ctx)

    # All three aligned frames must have the same row count = union of IDs.
    counts = {f.batch_size[0] for f in clip_t.frames}
    assert len(counts) == 1
    n = clip_t.frames[0].batch_size[0]
    assert n >= 2

    # The third frame has at least one row that's a Removed placeholder —
    # the tracklet that was filtered out of the raw snapshot.
    statuses_f2 = clip_t.frames[2].status.tolist()
    assert int(TrackletStatus.Removed) in statuses_f2


class _StubRefiner:
    """Asserts row-alignment of ClipTracklets across frames; returns input."""

    def __call__(self, clip_t, clip_ctx):
        del clip_ctx
        first = clip_t.frames[0].id
        for frame in clip_t.frames[1:]:
            assert torch.equal(frame.id, first)
        return clip_t


def test_refine_mode_runs_refiner_on_aligned_clip_tracklets():
    """NTH2 — ClipTracker(mode='refine') feeds aligned ClipTracklets to the refiner."""
    tr = _build()
    refiner = _StubRefiner()
    cliptr = ClipTracker(tr, mode="refine", refiner=refiner)

    frames = [
        _ds(torch.tensor([[1.0, 0.0]])),
        _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
    ]
    clip = ClipDetections(frames=frames)
    ctx = ClipFrameContext.make(start_frame=0, K=2, fps=15.0)
    snap_out, clip_t, _ = cliptr.process_clip(tr.empty_snapshot(), clip, ctx)
    # Stub refiner asserted alignment; if we get here it passed.
    assert clip_t.K == 2
    assert snap_out is clip_t.frames[-1]
