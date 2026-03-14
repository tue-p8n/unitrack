# tests/unitrack/data/test_gate.py
from __future__ import annotations

import pytest
import torch
from unitrack.data import CostExpression, Gate


def test_per_pair_construction():
    mask = torch.tensor([[True, False], [True, True]])
    g = Gate.PerPair(mask=mask)
    # Note: isinstance(g, Gate) is intentionally NOT checked —
    # Gate.PerPair resolves to _PerPair which is not a subclass of Gate.
    assert g.kind == "per_pair"
    assert torch.equal(g.mask, mask)


def test_per_pair_apply_to_cost_expression():
    mask = torch.tensor([[True, False], [True, True]])
    g = Gate.PerPair(mask=mask)
    base = CostExpression.from_matrix(torch.tensor([[0.1, 0.5], [0.4, 0.2]]))
    gated = g.apply(base)
    assert gated.gate_pair is not None
    assert torch.equal(gated.gate_pair, mask)


def test_combine_self_is_idempotent_for_per_pair():
    mask = torch.tensor([[True, False], [True, True]])
    g = Gate.PerPair(mask=mask)
    g2 = Gate.combine(g, g)
    assert torch.equal(g2.mask, mask)


def test_per_cs_apply_blocks_full_rows():
    keep = torch.tensor([True, False])
    g = Gate.PerCs(mask=keep)
    expr = CostExpression.from_matrix(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    out = g.apply(expr).materialize()
    assert out[0, 0].item() == pytest.approx(0.1, rel=1e-5)
    assert torch.isinf(out[1, 0])
    assert torch.isinf(out[1, 1])


def test_per_ds_apply_blocks_full_columns():
    keep = torch.tensor([True, False])
    g = Gate.PerDs(mask=keep)
    expr = CostExpression.from_matrix(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    out = g.apply(expr).materialize()
    assert torch.isinf(out[0, 1])
    assert torch.isinf(out[1, 1])
    assert out[0, 0].item() == pytest.approx(0.1, rel=1e-5)
    assert out[1, 0].item() == pytest.approx(0.3, rel=1e-5)


def test_cost_bias_apply_adds_to_matrix():
    bias = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    g = Gate.CostBias(matrix=bias)
    expr = CostExpression.from_matrix(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    out = g.apply(expr).materialize()
    assert out[0, 1].item() == pytest.approx(1.2, rel=1e-5)
    assert out[0, 0].item() == pytest.approx(0.1, rel=1e-5)


def test_combine_per_cs_and_per_ds_promotes_to_per_pair():
    cs_keep = torch.tensor([True, False])
    ds_keep = torch.tensor([True, True])
    g = Gate.combine(Gate.PerCs(mask=cs_keep), Gate.PerDs(mask=ds_keep))
    assert g.kind == "per_pair"
    expected = cs_keep[:, None] & ds_keep[None, :]
    assert torch.equal(g.mask, expected)


def test_combine_two_cost_biases_sums_them():
    a = Gate.CostBias(matrix=torch.tensor([[0.0, 1.0]]))
    b = Gate.CostBias(matrix=torch.tensor([[2.0, 0.0]]))
    g = Gate.combine(a, b)
    assert g.kind == "cost_bias"
    assert torch.equal(g.matrix, torch.tensor([[2.0, 1.0]]))


def test_combine_per_pair_and_cost_bias_yields_pair_with_attached_bias():
    pair = Gate.PerPair(mask=torch.tensor([[True, False]]))
    bias = Gate.CostBias(matrix=torch.tensor([[0.0, 0.5]]))
    g = Gate.combine(pair, bias)
    # Convention: combined object exposes both mask and bias_matrix; the
    # ``apply()`` of the result composes them onto the CostExpression.
    expr = CostExpression.from_matrix(torch.tensor([[1.0, 1.0]]))
    out = g.apply(expr)
    assert out.gate_pair is not None
    assert out.bias is not None
    assert torch.equal(out.gate_pair, torch.tensor([[True, False]]))
    assert torch.equal(out.bias, torch.tensor([[0.0, 0.5]]))


def test_combine_pair_and_bias_with_third_per_cs_remains_closed():
    # Regression for gate closure (C4): folding three operands where the
    # intermediate is _PairAndBias used to raise on the third combine.
    pair = Gate.PerPair(mask=torch.tensor([[True, True], [False, True]]))
    bias = Gate.CostBias(matrix=torch.tensor([[0.0, 0.5], [0.0, 0.0]]))
    cs_only = Gate.PerCs(mask=torch.tensor([True, True]))
    g12 = Gate.combine(pair, bias)  # _PairAndBias
    g123 = Gate.combine(g12, cs_only)  # must not crash
    expr = CostExpression.from_matrix(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    out = g123.apply(expr)
    # The combined object retains both the pair mask and the bias.
    assert out.gate_pair is not None
    assert out.bias is not None


def test_combine_pair_and_bias_with_cost_bias_sums_biases():
    pair = Gate.PerPair(mask=torch.tensor([[True, False]]))
    b1 = Gate.CostBias(matrix=torch.tensor([[0.0, 0.5]]))
    b2 = Gate.CostBias(matrix=torch.tensor([[0.0, 0.25]]))
    g12 = Gate.combine(pair, b1)  # _PairAndBias
    g123 = Gate.combine(g12, b2)  # closed: still has mask + summed bias
    expr = CostExpression.from_matrix(torch.tensor([[1.0, 1.0]]))
    out = g123.apply(expr)
    assert out.bias is not None
    assert torch.equal(out.bias, torch.tensor([[0.0, 0.75]]))
