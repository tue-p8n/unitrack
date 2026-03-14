"""Verify AVX2 and scalar lapjvs produce identical optimal assignments."""

from __future__ import annotations

import pytest
import scipy.optimize as sp
import torch
from unitrack.assignment.lapjv import lapjvs_assignment


def ref_cost(c):
    r, col = sp.linear_sum_assignment(c.numpy())
    return float(c.numpy()[r, col].sum())


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("seed", range(10))
def test_lapjvs_avx2_parity(n, seed):
    torch.manual_seed(seed)
    cost = torch.rand(n, n, dtype=torch.float64)
    m, _, _ = lapjvs_assignment(cost)
    got = cost[m[:, 0], m[:, 1]].sum().item()
    assert abs(got - ref_cost(cost)) < 1e-9, (
        f"n={n} seed={seed}: {got} vs {ref_cost(cost)}"
    )
