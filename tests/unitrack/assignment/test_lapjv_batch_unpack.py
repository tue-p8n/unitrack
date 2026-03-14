"""C++-unpacked batch results match the Python-unpacked reference."""

import pytest
import scipy.optimize as sp
import torch
from unitrack.assignment.lapjv import lapjvs_batch_assignment


def ref_solve(c):
    r, col = sp.linear_sum_assignment(c.numpy())
    return set(zip(r.tolist(), col.tolist(), strict=True))


@pytest.mark.parametrize(("batch_size", "n"), [(8, 32), (4, 128), (2, 256)])
def test_batch_unpack_parity(batch_size, n):
    torch.manual_seed(42)
    costs = [torch.rand(n, n, dtype=torch.float64) for _ in range(batch_size)]
    results = lapjvs_batch_assignment(costs)
    for i, (m, _ur, _uc) in enumerate(results):
        got = {(m[k, 0].item(), m[k, 1].item()) for k in range(m.shape[0])}
        exp = ref_solve(costs[i])
        assert got == exp, f"batch_size={batch_size} n={n} problem {i}: mismatch"
