# tests/unitrack/costs/test_chamfer.py
from __future__ import annotations

import torch
from unitrack.costs import Chamfer
from unitrack.data import Detections, FrameContext, Tracklets


def test_identical_point_clouds_give_zero_chamfer():
    n, m = 1, 1
    cloud = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    cs = Tracklets(
        id=torch.zeros(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        cloud=cloud,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.zeros(m, dtype=torch.int64), cloud=cloud, batch_size=[m]
    )
    cost = Chamfer("cloud")(cs, ds, FrameContext.make(0)).matrix
    assert cost.item() < 1e-6
