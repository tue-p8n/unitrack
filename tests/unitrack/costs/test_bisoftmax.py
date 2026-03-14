# tests/unitrack/costs/test_bisoftmax.py
from __future__ import annotations

import torch
from unitrack.costs import BiSoftmax
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


def test_bisoftmax_identical_vectors_smaller_cost_than_orthogonal():
    # Need multiple entries so softmax is non-trivial.
    # Row 0 is the pair under test; row 1 is a distractor.
    cs_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ds_same = torch.tensor([[1.0, 0.0], [0.5, 0.5]])  # col 0 matches row 0
    ds_orth = torch.tensor([[0.0, 1.0], [0.5, 0.5]])  # col 0 orthogonal to row 0
    cs, ds1, ctx = _make(cs_feat, ds_same)
    _, ds2, _ = _make(cs_feat, ds_orth)
    cost_same = BiSoftmax("feat")(cs, ds1, ctx).matrix[0, 0].item()
    cost_orth = BiSoftmax("feat")(cs, ds2, ctx).matrix[0, 0].item()
    assert cost_same < cost_orth


def test_bisoftmax_output_shape():
    n, m = 3, 4
    cs_feat = torch.randn(n, 8)
    ds_feat = torch.randn(m, 8)
    cs, ds, ctx = _make(cs_feat, ds_feat)
    cost = BiSoftmax("feat")(cs, ds, ctx).matrix
    assert cost.shape == (n, m)
