# tests/unitrack/costs/test_rbf.py
from __future__ import annotations

import torch
from unitrack.costs import RBF
from unitrack.data import Detections, FrameContext, Tracklets


def _make(cs_feat: torch.Tensor, ds_feat: torch.Tensor):
    n, m = cs_feat.shape[0], ds_feat.shape[0]
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        feat=cs_feat,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64), feat=ds_feat, batch_size=[m]
    )
    return cs, ds, FrameContext.make(0)


def test_rbf_identical_vectors_give_zero():
    v = torch.tensor([[1.0, 0.0, 0.0]])
    cs, ds, ctx = _make(v, v)
    cost = RBF("feat")(cs, ds, ctx).matrix
    assert cost.item() < 1e-6


def test_rbf_distant_vectors_approach_one():
    cs_feat = torch.zeros(1, 3)
    ds_feat = torch.tensor([[100.0, 100.0, 100.0]])
    cs, ds, ctx = _make(cs_feat, ds_feat)
    cost = RBF("feat", gamma=1.0)(cs, ds, ctx).matrix
    assert abs(cost.item() - 1.0) < 1e-5


def test_rbf_output_in_range():
    n, m = 3, 4
    cs_feat = torch.randn(n, 8)
    ds_feat = torch.randn(m, 8)
    cs, ds, ctx = _make(cs_feat, ds_feat)
    cost = RBF("feat")(cs, ds, ctx).matrix
    assert cost.shape == (n, m)
    assert (cost >= 0).all()
    assert (cost <= 1).all()
