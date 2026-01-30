r"""Tests for ``unitrack.costs``."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tensordict import TensorDict
from unitrack import costs

KEY_GATED_FIELD = "gated_field"


@settings(deadline=None)
@given(cs_num=st.integers(0, 4), ds_num=st.integers(0, 4))
def test_gate_cost(cs_num, ds_num):
    cs = TensorDict.from_dict(
        {KEY_GATED_FIELD: torch.arange(cs_num)}, batch_size=[cs_num]
    )
    assert len(cs) == cs_num, cs

    ds = TensorDict.from_dict(
        {KEY_GATED_FIELD: torch.arange(ds_num)}, batch_size=[ds_num]
    )
    assert len(ds) == ds_num, ds

    gate = costs.GateCost(KEY_GATED_FIELD)

    gate_matrix: torch.Tensor = gate(cs, ds)
    gate_shape = gate_matrix.shape
    assert gate_matrix.shape[0] == cs_num, gate_shape
    assert gate_matrix.shape[1] == ds_num, gate_shape

    gate_mask = torch.full_like(gate_matrix, False, dtype=torch.bool)
    gate_mask.fill_diagonal_(True)

    assert torch.all(gate_matrix[gate_mask] == 0.0)
    assert torch.all(gate_matrix[~gate_mask] == torch.inf)


KEY_VALUE = "foo_bar"


@settings(deadline=None)
@given(cs_num=st.integers(0, 10), ds_num=st.integers(0, 10))
def test_cdist_cost(cs_num: int, ds_num: int):
    x_cs = cs_num * 2.0
    cs = TensorDict.from_dict(
        {KEY_VALUE: torch.arange(cs_num).unsqueeze(1).float() * x_cs},
        batch_size=[cs_num],
    )
    assert len(cs) == cs_num, cs
    x_ds = ds_num * 3.0
    ds = TensorDict.from_dict(
        {KEY_VALUE: torch.arange(ds_num).unsqueeze(1).float() * x_ds},
        batch_size=[ds_num],
    )
    assert len(ds) == ds_num, ds

    cost = costs.CDist(KEY_VALUE)

    cost_matrix = cost(cs, ds)
    cost_shape = cost_matrix.shape
    assert cost_shape == (len(cs), len(ds)), cost_shape

    for cs_i in range(cs_num):
        for ds_i in range(ds_num):
            euclidean_distance = abs((cs_i * x_cs) - (ds_i * x_ds))
            assert cost_matrix[cs_i, ds_i] == euclidean_distance


KEY_MASK = "my_mask"


def test_mask_iou():
    cs_x = torch.tensor(
        [
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ],
        dtype=torch.int,
    )
    cs = TensorDict.from_dict({KEY_MASK: cs_x}, batch_size=[3])

    ds_x = torch.tensor(
        [
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ],
        dtype=torch.int,
    )
    ds = TensorDict.from_dict({KEY_MASK: ds_x}, batch_size=[2])

    cost = costs.MaskIoU(KEY_MASK)
    cost_matrix = cost(cs, ds)
    cost_shape = cost_matrix.shape

    assert cost_shape == (len(cs), len(ds)), cost_shape

    print(cost_matrix)