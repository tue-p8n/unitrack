# tests/unitrack/data/test_clip.py
from __future__ import annotations

import torch
from unitrack.data import (
    ClipDetections,
    ClipFrameContext,
    Detections,
)


def test_clip_detections_holds_per_frame_lists():
    frames = [
        Detections(
            index=torch.arange(3, dtype=torch.int64),
            kernel=torch.zeros(3, 4),
            batch_size=[3],
        ),
        Detections(
            index=torch.arange(2, dtype=torch.int64),
            kernel=torch.zeros(2, 4),
            batch_size=[2],
        ),
    ]
    clip = ClipDetections(frames=frames)
    assert clip.K == 2
    assert clip.frames[0].batch_size[0] == 3
    assert clip.frames[1].batch_size[0] == 2


def test_clip_frame_context_default_make():
    ctx = ClipFrameContext.make(start_frame=10, K=3, fps=15.0, stream_key=0)
    assert ctx.K == 3
    assert ctx.frame_contexts[0].frame_idx.item() == 10
    assert ctx.frame_contexts[2].frame_idx.item() == 12


def test_clip_detections_stacked_rejects_existing_frame_idx():
    """``stacked()`` would silently overwrite a caller-supplied ``frame_idx``
    user field; reject up-front with a clear error."""
    import pytest

    frames = [
        Detections(
            index=torch.arange(2, dtype=torch.int64),
            kernel=torch.zeros(2, 4),
            frame_idx=torch.zeros(2, dtype=torch.int64),
            batch_size=[2],
        ),
    ]
    with pytest.raises(ValueError, match="frame_idx"):
        ClipDetections(frames=frames).stacked()


def test_clip_detections_stacked_empty_clip_has_frame_idx_column():
    """The empty-clip path must keep a ``frame_idx`` column so downstream
    consumers don't see a ragged schema vs the non-empty case."""
    out = ClipDetections(frames=[]).stacked()
    assert "frame_idx" in out
    assert out["frame_idx"].shape == (0,)
