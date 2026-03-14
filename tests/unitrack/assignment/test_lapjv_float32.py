"""float32-native dispatch produces optimal or near-optimal assignments."""

import pytest
import scipy.optimize as sp
import torch
from unitrack.assignment import lapjv


def ref_cost(c_f64):
    r, col = sp.linear_sum_assignment(c_f64.numpy())
    return float(c_f64.numpy()[r, col].sum())


@pytest.mark.parametrize("n", [8, 32, 64, 128, 256])
@pytest.mark.parametrize("fn_name", ["lapjvx_assignment", "lapjvs_assignment"])
def test_float32_native_optimal(n, fn_name):
    torch.manual_seed(0)
    cost_f32 = torch.rand(n, n, dtype=torch.float32)
    cost_f64 = cost_f32.to(torch.float64)
    fn = getattr(lapjv, fn_name)
    m, _, _ = fn(cost_f32)
    got = cost_f64[m[:, 0], m[:, 1]].sum().item()
    assert abs(got - ref_cost(cost_f64)) / max(ref_cost(cost_f64), 1e-9) < 0.01
