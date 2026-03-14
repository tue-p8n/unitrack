# tests/unitrack/costs/test_cosine.py
from __future__ import annotations

import torch
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, Tracklets


def _make_inputs(cs_kernel: torch.Tensor, ds_kernel: torch.Tensor):
    n = cs_kernel.shape[0]
    m = ds_kernel.shape[0]
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        kernel=cs_kernel,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=ds_kernel, batch_size=[m]
    )
    ctx = FrameContext.make(0, delta=0.0)
    return cs, ds, ctx


def test_identical_vectors_give_zero_cost():
    cs_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ds_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cs, ds, ctx = _make_inputs(cs_k, ds_k)
    cost = Cosine("kernel")(cs, ds, ctx)
    assert torch.allclose(cost.matrix.diag(), torch.zeros(2), atol=1e-6)


def test_orthogonal_vectors_give_unit_cost():
    cs_k = torch.tensor([[1.0, 0.0]])
    ds_k = torch.tensor([[0.0, 1.0]])
    cs, ds, ctx = _make_inputs(cs_k, ds_k)
    cost = Cosine("kernel")(cs, ds, ctx)
    assert torch.allclose(cost.matrix, torch.tensor([[1.0]]), atol=1e-6)


def test_zero_input_handled_via_eps():
    cs_k = torch.zeros(1, 4)
    ds_k = torch.zeros(1, 4)
    cs, ds, ctx = _make_inputs(cs_k, ds_k)
    cost = Cosine("kernel", eps=1e-6)(cs, ds, ctx)
    assert torch.isfinite(cost.matrix).all()


def test_required_field_missing_raises():
    import pytest

    cs_k = torch.zeros(1, 4)
    ds_k = torch.zeros(1, 4)
    cs, ds, ctx = _make_inputs(cs_k, ds_k)
    with pytest.raises(KeyError, match="ghost"):
        Cosine("ghost")(cs, ds, ctx)
