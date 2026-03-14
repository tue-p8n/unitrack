from __future__ import annotations

import torch
from unitrack.data import FrameContext, Tracklets
from unitrack.states.kalman import KalmanCentroid2D


def _make(mean):
    n, d = mean.shape
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        centroid=mean,
        centroid_cov=torch.eye(2 * d).expand(n, 2 * d, 2 * d).contiguous(),
        batch_size=[n],
    )


def test_centroid_2d_forward_in_x_with_velocity():
    mean = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # x, y, vx, vy
    snap = _make(mean[:, :2])
    snap = snap.set("centroid", mean)
    proc = KalmanCentroid2D()
    out = proc(snap, FrameContext.make(0, delta=2.0))
    assert torch.allclose(out.centroid[0, :2], torch.tensor([2.0, 0.0]))
