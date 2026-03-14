# tests/unitrack/costs/test_combinators.py
from __future__ import annotations

import torch
from unitrack.costs import CDist, Cosine, Reduce, Weighted
from unitrack.data import Detections, FrameContext, Tracklets


def _pair(kernel_cs, kernel_ds, pos_cs, pos_ds):
    n, m = kernel_cs.shape[0], kernel_ds.shape[0]
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        kernel=kernel_cs,
        position=pos_cs,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64),
        kernel=kernel_ds,
        position=pos_ds,
        batch_size=[m],
    )
    return cs, ds, FrameContext.make(0)


def test_weighted_scales_inner_cost():
    cs, ds, ctx = _pair(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
    )
    inner = Cosine("kernel")
    weighted = Weighted(inner, 3.0)(cs, ds, ctx).matrix
    plain = inner(cs, ds, ctx).matrix
    assert torch.allclose(weighted, 3.0 * plain)


def test_reduce_sum_combines_two_children():
    cs_k = torch.tensor([[1.0, 0.0]])
    ds_k = torch.tensor([[1.0, 0.0]])
    cs_p = torch.tensor([[0.0, 0.0]])
    ds_p = torch.tensor([[1.0, 0.0]])
    cs, ds, ctx = _pair(cs_k, ds_k, cs_p, ds_p)
    out = Reduce([Cosine("kernel"), CDist("position")], "sum")(cs, ds, ctx).matrix
    expected = (
        Cosine("kernel")(cs, ds, ctx).matrix + CDist("position")(cs, ds, ctx).matrix
    )
    assert torch.allclose(out, expected)


def test_reduce_min_chooses_pointwise_minimum():
    cs_k = torch.tensor([[1.0, 0.0]])
    ds_k = torch.tensor([[0.0, 1.0]])  # large cosine
    cs_p = torch.tensor([[0.0, 0.0]])
    ds_p = torch.tensor([[0.1, 0.0]])  # small cdist
    cs, ds, ctx = _pair(cs_k, ds_k, cs_p, ds_p)
    out = Reduce([Cosine("kernel"), CDist("position")], "min")(cs, ds, ctx).matrix
    a = Cosine("kernel")(cs, ds, ctx).matrix
    b = CDist("position")(cs, ds, ctx).matrix
    assert torch.allclose(out, torch.minimum(a, b))
