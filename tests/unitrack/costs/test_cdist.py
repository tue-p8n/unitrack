# tests/unitrack/costs/test_cdist.py
from __future__ import annotations

import torch
from unitrack.costs import CDist
from unitrack.data import Detections, FrameContext, Tracklets


def _make(cs_pos: torch.Tensor, ds_pos: torch.Tensor):
    n, m = cs_pos.shape[0], ds_pos.shape[0]
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        position=cs_pos,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64), position=ds_pos, batch_size=[m]
    )
    return cs, ds, FrameContext.make(0)


def test_cdist_l2():
    cs_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    ds_pos = torch.tensor([[0.0, 1.0], [3.0, 4.0]])
    cs, ds, ctx = _make(cs_pos, ds_pos)
    cost = CDist("position", p_norm=2.0)(cs, ds, ctx).matrix
    sqrt2 = torch.sqrt(torch.tensor(2.0))
    sqrt20 = torch.sqrt(torch.tensor(20.0))
    expected = torch.tensor([[1.0, 5.0], [sqrt2, sqrt20]])
    assert torch.allclose(cost, expected, atol=1e-5)


def test_cdist_l1():
    cs_pos = torch.tensor([[0.0, 0.0]])
    ds_pos = torch.tensor([[3.0, 4.0]])
    cs, ds, ctx = _make(cs_pos, ds_pos)
    cost = CDist("position", p_norm=1.0)(cs, ds, ctx).matrix
    assert cost.item() == 7.0


def test_cdist_required_field_missing():
    import pytest

    cs, ds, ctx = _make(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(KeyError, match="ghost"):
        CDist("ghost")(cs, ds, ctx)
