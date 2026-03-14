# tests/unitrack/gates/test_simple.py
from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.gates import ClassGate, NoneGate, ScoreGate


def _pair(cs_field: dict, ds_field: dict, n: int, m: int) -> tuple:
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        **cs_field,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64), **ds_field, batch_size=[m]
    )
    return cs, ds, FrameContext.make(0)


def test_none_gate_returns_all_true_per_pair():
    cs, ds, ctx = _pair({}, {}, n=3, m=2)
    g = NoneGate()(cs, ds, ctx)
    assert g.kind == "per_pair"
    assert g.mask.shape == (3, 2)
    assert torch.all(g.mask)


def test_class_gate_outer_equal():
    cs, ds, ctx = _pair(
        {"klass": torch.tensor([0, 1, 0], dtype=torch.int64)},
        {"klass": torch.tensor([0, 1], dtype=torch.int64)},
        n=3,
        m=2,
    )
    g = ClassGate("klass")(cs, ds, ctx)
    expected = torch.tensor([[True, False], [False, True], [True, False]])
    assert torch.equal(g.mask, expected)


def test_score_gate_returns_per_ds():
    cs, ds, ctx = _pair({}, {"score": torch.tensor([0.4, 0.6, 0.8])}, n=2, m=3)
    g = ScoreGate("score", threshold=0.5)(cs, ds, ctx)
    assert g.kind == "per_ds"
    assert g.mask.tolist() == [False, True, True]
