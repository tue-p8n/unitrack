# tests/unitrack/data/test_cost.py
from __future__ import annotations

import pytest
import torch
from unitrack.data import CostExpression


def test_minimal_cost_no_gates():
    matrix = torch.tensor([[0.1, 0.5], [0.4, 0.2]])
    c = CostExpression.from_matrix(matrix)
    assert torch.equal(c.matrix, matrix)
    assert c.gate_pair is None
    assert c.gate_cs is None
    assert c.gate_ds is None
    assert c.bias is None


def test_apply_pair_gate_replaces_blocked_with_inf():
    matrix = torch.tensor([[0.1, 0.5], [0.4, 0.2]])
    pair = torch.tensor([[True, False], [True, True]])
    c = CostExpression.from_matrix(matrix, gate_pair=pair)
    out = c.materialize()
    assert torch.isinf(out[0, 1])
    assert out[0, 0].item() == pytest.approx(0.1, rel=1e-5)
    assert out[1, 0].item() == pytest.approx(0.4, rel=1e-5)
    assert out[1, 1].item() == pytest.approx(0.2, rel=1e-5)


def test_materialize_with_bias_adds_to_matrix():
    matrix = torch.tensor([[0.1, 0.5], [0.4, 0.2]])
    bias = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    c = CostExpression.from_matrix(matrix, bias=bias)
    out = c.materialize()
    assert out[0, 1].item() == pytest.approx(1.5, rel=1e-5)
    assert out[0, 0].item() == pytest.approx(0.1, rel=1e-5)


def test_per_side_gates_block_full_rows_and_columns():
    matrix = torch.tensor([[0.1, 0.5], [0.4, 0.2]])
    cs_keep = torch.tensor([True, False])
    ds_keep = torch.tensor([True, True])
    c = CostExpression.from_matrix(matrix, gate_cs=cs_keep, gate_ds=ds_keep)
    out = c.materialize()
    assert torch.isinf(out[1, 0])
    assert torch.isinf(out[1, 1])
    assert out[0, 0].item() == pytest.approx(0.1, rel=1e-5)
    assert out[0, 1].item() == pytest.approx(0.5, rel=1e-5)


def test_from_matrix_rejects_non_2d_matrix():
    with pytest.raises(ValueError, match="must be 2-D"):
        CostExpression.from_matrix(torch.tensor([0.1, 0.5]))


def test_from_matrix_rejects_mismatched_gate_pair():
    matrix = torch.zeros(2, 3)
    bad = torch.zeros(2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="gate_pair shape"):
        CostExpression.from_matrix(matrix, gate_pair=bad)


def test_from_matrix_rejects_mismatched_gate_cs():
    matrix = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="gate_cs shape"):
        CostExpression.from_matrix(matrix, gate_cs=torch.zeros(3, dtype=torch.bool))


def test_from_matrix_rejects_mismatched_gate_ds():
    matrix = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="gate_ds shape"):
        CostExpression.from_matrix(matrix, gate_ds=torch.zeros(2, dtype=torch.bool))


def test_from_matrix_rejects_mismatched_bias():
    matrix = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="bias shape"):
        CostExpression.from_matrix(matrix, bias=torch.zeros(3, 2))
