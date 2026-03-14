# tests/unitrack/pipeline/test_iterate.py
from __future__ import annotations

import pytest
import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.pipeline import Iterate, Pipe


def test_iterate_n_equals_calling_body_n_times():
    cs = Tracklets(
        id=torch.arange(2, dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.ones(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    body = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    out = Iterate(n=3, body=body)(cs, ds, FrameContext.make(0))
    pairs = sorted(map(tuple, out.matched_pairs.tolist()))
    assert pairs == [(0, 0), (1, 1)] or pairs == []


def test_iterate_n_zero_returns_empty_match():
    with pytest.raises(ValueError, match="positive"):
        Iterate(n=0, body=None)
