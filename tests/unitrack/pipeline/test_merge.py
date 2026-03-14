# tests/unitrack/pipeline/test_merge.py
from __future__ import annotations

import torch
from unitrack.data import CostExpression
from unitrack.pipeline.merge import Max, Mean, Min, WeightedSum


def test_weighted_sum():
    a = CostExpression.from_matrix(torch.tensor([[1.0, 2.0]]))
    b = CostExpression.from_matrix(torch.tensor([[3.0, 4.0]]))
    out = WeightedSum([2.0, 0.5])([a, b])
    assert torch.allclose(out.matrix, torch.tensor([[3.5, 6.0]]))


def test_min_pointwise():
    a = CostExpression.from_matrix(torch.tensor([[1.0, 5.0]]))
    b = CostExpression.from_matrix(torch.tensor([[3.0, 2.0]]))
    out = Min()([a, b])
    assert torch.equal(out.matrix, torch.tensor([[1.0, 2.0]]))


def test_max_pointwise():
    a = CostExpression.from_matrix(torch.tensor([[1.0, 5.0]]))
    b = CostExpression.from_matrix(torch.tensor([[3.0, 2.0]]))
    out = Max()([a, b])
    assert torch.equal(out.matrix, torch.tensor([[3.0, 5.0]]))


def test_mean_pointwise():
    a = CostExpression.from_matrix(torch.tensor([[1.0, 5.0]]))
    b = CostExpression.from_matrix(torch.tensor([[3.0, 1.0]]))
    out = Mean()([a, b])
    assert torch.equal(out.matrix, torch.tensor([[2.0, 3.0]]))


def test_weighted_sum_with_one_branch_gated_does_not_crash_on_tensor_or():
    # Regression for C3: Python ``or`` on a multi-element bool tensor used
    # to raise ``Boolean value of Tensor with more than one element is
    # ambiguous`` whenever exactly one branch carried a gate slot.
    pair = torch.tensor([[True, True], [True, False]])
    a = CostExpression.from_matrix(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        gate_pair=pair,
    )
    b = CostExpression.from_matrix(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    out = WeightedSum([1.0, 1.0])([a, b])
    assert out.gate_pair is not None
    assert torch.equal(out.gate_pair, pair)


def test_weighted_sum_two_branches_with_gates_ands_them():
    pair_a = torch.tensor([[True, True], [True, False]])
    pair_b = torch.tensor([[True, False], [True, True]])
    a = CostExpression.from_matrix(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), gate_pair=pair_a
    )
    b = CostExpression.from_matrix(
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]), gate_pair=pair_b
    )
    out = WeightedSum([1.0, 1.0])([a, b])
    assert out.gate_pair is not None
    assert torch.equal(out.gate_pair, pair_a & pair_b)


def test_weighted_sum_one_branch_biased_does_not_crash():
    bias = torch.tensor([[0.0, 0.5], [0.0, 0.0]])
    a = CostExpression.from_matrix(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), bias=bias)
    b = CostExpression.from_matrix(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    out = WeightedSum([1.0, 1.0])([a, b])
    assert out.bias is not None
    assert torch.equal(out.bias, bias)
