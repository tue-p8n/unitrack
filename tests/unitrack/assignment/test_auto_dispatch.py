r"""Tests for the smart-dispatch LAP entry points.

Covers:
  * :func:`unitrack.assignment.auto_assignment` and
    :func:`unitrack.assignment.auto_batch_assignment` route to the CPU
    LAPJV path by default and to the torchmatch CUDA solver when
    ``prefer="cuda"`` is set.
  * :class:`unitrack.assignment.AutoLAP` integrates with the
    :class:`Assignment` ``nn.Module`` interface.
"""

from __future__ import annotations

import pytest
import torch
from unitrack import assignment
from unitrack.assignment import AutoLAP, Prefer, auto_assignment, auto_batch_assignment

_CUDA_AVAILABLE = torch.cuda.is_available()
_CUDA_SKIP = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")


@pytest.fixture
def cost_cpu() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(8, 8, dtype=torch.float32)


@pytest.fixture
def cost_cuda() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(8, 8, dtype=torch.float32, device="cuda")


def test_auto_assignment_default_routes_cpu(cost_cpu):
    matches, ur, _uc = auto_assignment(cost_cpu)
    assert matches.device.type == "cpu"
    assert matches.shape[1] == 2
    assert matches.shape[0] + ur.shape[0] == cost_cpu.shape[0]


@_CUDA_SKIP
def test_auto_assignment_cpu_path_with_cuda_input(cost_cuda):
    matches, ur, _uc = auto_assignment(cost_cuda, prefer=Prefer.CPU)
    # Result lives on the input device after the round-trip.
    assert matches.device.type == "cuda"
    assert matches.shape[0] + ur.shape[0] == cost_cuda.shape[0]


@_CUDA_SKIP
def test_auto_assignment_cuda_pin(cost_cuda):
    matches, ur, _uc = auto_assignment(cost_cuda, prefer=Prefer.CUDA)
    assert matches.device.type == "cuda"
    assert matches.shape[0] + ur.shape[0] == cost_cuda.shape[0]


@pytest.mark.parametrize("prefer", [Prefer.AUTO, Prefer.CPU, "auto", "cpu"])
def test_auto_batch_assignment_default(cost_cpu, prefer):
    out = auto_batch_assignment([cost_cpu, cost_cpu * 2.0], prefer=prefer)
    assert len(out) == 2
    for matches, _ur, _uc in out:
        assert matches.device.type == "cpu"
        assert matches.shape[1] == 2


@_CUDA_SKIP
def test_auto_batch_assignment_cuda_pin(cost_cuda):
    out = auto_batch_assignment([cost_cuda, cost_cuda * 2.0], prefer=Prefer.CUDA)
    assert len(out) == 2
    for matches, _ur, _uc in out:
        assert matches.device.type == "cuda"


def test_auto_lap_module_default(cost_cpu):
    mod = AutoLAP()
    matches, _ur, _uc = mod(cost_cpu)
    assert matches.device.type == "cpu"
    assert isinstance(mod, assignment.Assignment)


@_CUDA_SKIP
def test_auto_lap_module_cuda(cost_cuda):
    mod = AutoLAP(prefer=Prefer.CUDA)
    matches, _ur, _uc = mod(cost_cuda)
    assert matches.device.type == "cuda"


# ---------------------------------------------------------------------------
# Dtype coverage: lapjvx upcasts non-native dtypes via _solver_dtype();
# auto_assignment must accept the full floating range on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.float32, torch.float16, torch.bfloat16],
)
def test_lapjv_assignment_dtype_coverage(dtype):
    """CPU LAPJV upcasts all floating dtypes via _solver_dtype()."""
    torch.manual_seed(0)
    cost = (torch.rand(8, 8) * 100.0).to(dtype=dtype)
    matches, ur, _uc = auto_assignment(cost)
    assert matches.shape[0] + ur.shape[0] == 8


@_CUDA_SKIP
@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.float32, torch.float16, torch.bfloat16],
)
def test_lap_assignment_dtype_coverage_cuda(dtype):
    """CUDA lap_assignment upcasts via .to(dtype=torch.float32)."""
    from unitrack.assignment.lap import lap_assignment

    torch.manual_seed(0)
    cost = (torch.rand(8, 8, device="cuda") * 100.0).to(dtype=dtype)
    matches, ur, _uc = lap_assignment(cost)
    assert matches.shape[0] + ur.shape[0] == 8


@_CUDA_SKIP
def test_lap_assignment_respects_input_device():
    """``lap_assignment`` must route through the input tensor's device."""
    from unitrack.assignment.lap import lap_assignment

    torch.manual_seed(0)
    cost = torch.rand(8, 8, dtype=torch.float32, device="cuda:0")
    matches, ur, _uc = lap_assignment(cost)
    assert matches.device == cost.device
    assert ur.device == cost.device
    assert matches.shape[0] + ur.shape[0] == cost.shape[0]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    reason="needs >=2 CUDA devices",
)
def test_lap_assignment_multi_gpu():
    """Solves on cuda:0 and cuda:1 hit independent device routing."""
    from unitrack.assignment.lap import lap_assignment

    torch.manual_seed(0)
    for dev in ("cuda:0", "cuda:1"):
        cost = torch.rand(16, 16, dtype=torch.float32, device=dev)
        matches, _ur, _uc = lap_assignment(cost)
        assert matches.device == torch.device(dev)
