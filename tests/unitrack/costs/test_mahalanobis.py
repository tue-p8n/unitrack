# tests/unitrack/costs/test_mahalanobis.py
from __future__ import annotations

import torch
from unitrack.costs import Mahalanobis
from unitrack.data import Detections, FrameContext, Tracklets


def test_mahalanobis_isotropic_unit_cov_matches_squared_l2():
    n, m, d = 2, 2, 3
    cs_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cs_cov = torch.eye(d).expand(n, d, d)
    ds_pos = torch.tensor([[0.0, 1.0, 0.0], [3.0, 4.0, 0.0]])
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        position=cs_pos,
        position_cov=cs_cov,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64),
        position=ds_pos,
        batch_size=[m],
    )
    cost = Mahalanobis("position", "position_cov")(cs, ds, FrameContext.make(0)).matrix
    sq_l2 = torch.cdist(cs_pos, ds_pos, p=2.0) ** 2
    assert torch.allclose(cost, sq_l2, atol=1e-5)
