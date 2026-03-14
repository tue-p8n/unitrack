"""lapjvx flat-pointer refactor produces identical results to original."""

import pytest
import scipy.optimize as sp
import torch
from unitrack.assignment.lapjv import lapjvx_assignment


def ref_cost(c):
    r, col = sp.linear_sum_assignment(c.numpy())
    return float(c.numpy()[r, col].sum())


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("seed", range(10))
def test_lapjvx_flat_parity(n, seed):
    torch.manual_seed(seed)
    cost = torch.rand(n, n, dtype=torch.float64)
    m, _, _ = lapjvx_assignment(cost)
    got = cost[m[:, 0], m[:, 1]].sum().item()
    assert abs(got - ref_cost(cost)) < 1e-9, f"n={n} seed={seed}"
