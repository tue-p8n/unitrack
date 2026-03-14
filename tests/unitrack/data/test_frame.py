# tests/unitrack/data/test_frame.py
from __future__ import annotations

import torch
from unitrack.data import FrameContext


def test_construct_minimal():
    ctx = FrameContext(
        frame_idx=torch.tensor(7, dtype=torch.int64),
        delta=torch.tensor(0.066, dtype=torch.float32),
        fps=torch.tensor(15.0, dtype=torch.float32),
        stream_key=torch.tensor(0, dtype=torch.int64),
        batch_size=[],
    )
    assert ctx.frame_idx.item() == 7
    assert _approx_eq(ctx.delta.item(), 0.066)
    assert ctx.fps.item() == 15.0


def _approx_eq(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) < eps


def test_make_helper_from_python_scalars():
    ctx = FrameContext.make(frame_idx=3, delta=0.1, fps=10.0, stream_key=42)
    assert ctx.frame_idx.item() == 3
    assert ctx.stream_key.item() == 42
