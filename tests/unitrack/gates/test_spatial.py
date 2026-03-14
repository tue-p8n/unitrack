# tests/unitrack/gates/test_spatial.py
from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.gates import SpatialGate2D, SpatialGate3D


def test_spatial_gate_2d_pair_mask():
    n, m = 2, 2
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        position=torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64),
        position=torch.tensor([[0.5, 0.0], [10.5, 0.0]]),
        batch_size=[m],
    )
    g = SpatialGate2D("position", max_dist=2.0)(cs, ds, FrameContext.make(0))
    expected = torch.tensor([[True, False], [False, True]])
    assert torch.equal(g.mask, expected)


def test_spatial_gate_3d_uses_3d_distance():
    cs = Tracklets(
        id=torch.zeros(1, dtype=torch.int64),
        status=torch.ones(1, dtype=torch.int8),
        hits=torch.ones(1, dtype=torch.int32),
        time_since_update=torch.zeros(1, dtype=torch.int32),
        age=torch.ones(1, dtype=torch.int32),
        frame_started=torch.zeros(1, dtype=torch.int32),
        frame_last_seen=torch.zeros(1, dtype=torch.int32),
        position=torch.tensor([[0.0, 0.0, 0.0]]),
        batch_size=[1],
    )
    ds = Detections(
        index=torch.zeros(1, dtype=torch.int64),
        position=torch.tensor([[3.0, 4.0, 12.0]]),  # distance = 13
        batch_size=[1],
    )
    g_in = SpatialGate3D("position", max_dist=14.0)(cs, ds, FrameContext.make(0))
    g_out = SpatialGate3D("position", max_dist=12.0)(cs, ds, FrameContext.make(0))
    assert g_in.mask.item()
    assert not g_out.mask.item()
