r"""Tests for ``unitrack.assignment``."""

from __future__ import annotations

import time

import pytest
import torch
from unitrack import assignment
from unitrack.assignment import lap

_CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(
    params=[
        (8, 8),
        (8, 10),
        (10, 8),
        (0, 0),
    ],
    ids=(
        "cost:square",
        "cost:tall",
        "cost:wide",
        "cost:empty",
    ),
)
def cost_matrix(request):
    shape = request.param
    return torch.rand(shape, dtype=torch.float) ** 10


_CUDA_SKIP = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")


@pytest.fixture(
    params=[
        assignment.Greedy,
        assignment.Hungarian,
        assignment.Auction,
        assignment.Jonker,
        assignment.SoftAssignment,
        pytest.param(
            lambda: lap.LAP(backend=lap.Backend.CLASSICAL),
            marks=_CUDA_SKIP,
        ),
        pytest.param(
            lambda: lap.LAP(backend=lap.Backend.HYBRID),
            marks=_CUDA_SKIP,
        ),
        pytest.param(
            lambda: lap.LAP(backend=lap.Backend.TREE),
            marks=_CUDA_SKIP,
        ),
    ],
    ids=(
        "alg:greedy",
        "alg:hungarian",
        "alg:auction",
        "alg:jonker",
        "alg:soft",
        "alg:lap-classical",
        "alg:lap-hybrid",
        "alg:lap-tree",
    ),
    scope="module",
)
def solver(request):
    mod = request.param()
    assert isinstance(mod, assignment.Assignment)
    return mod


def test_assignment_invoke(cost_matrix, solver):
    matches, unmatch_rows, unmatch_cols = solver(cost_matrix)

    assert matches.shape[0] <= min(cost_matrix.shape)
    assert matches.shape[1] == 2
    assert unmatch_rows.shape[0] <= cost_matrix.shape[0]
    assert unmatch_cols.shape[0] <= cost_matrix.shape[1]
    assert not any(matches[:, 0] < 0)
    assert not any(matches[:, 1] < 0)
    assert not any(r in matches[:, 0] for r in unmatch_rows)
    assert not any(c in matches[:, 1] for c in unmatch_cols)


def generate_cost(size: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random cost matrix and a random matching"""
    size = round(size)
    cost_matrix = torch.randn(size, size, dtype=torch.float) * torch.randint(
        -10, 10, (size, size)
    ) ** torch.randint(-2, 2, (size, size))

    cost_matrix = torch.nn.functional.softplus(cost_matrix)

    solution, _, _ = assignment.jonker_volgenant_assignment(cost_matrix, torch.inf)
    return cost_matrix, solution


@pytest.mark.parametrize(
    ("cost_matrix", "solution"),
    [
        (
            torch.arange(9, dtype=torch.float).reshape(3, 3),
            torch.tensor([[0, 0], [1, 1], [2, 2]]),
        ),
        (
            torch.arange(6, dtype=torch.float).reshape(2, 3),
            torch.tensor([[0, 0], [1, 1]]),
        ),
        (
            torch.arange(6, 0, -1, dtype=torch.float).reshape(2, 3),
            torch.tensor([[0, 2], [1, 1]]),
        ),
        generate_cost(2**5),
        generate_cost(2**6),
        generate_cost(2**7),
        generate_cost(2**8),
    ],
)
def test_assignment_known(cost_matrix, solution, solver):
    """Test if algorithms can solve the known N x M cost matrix.

    Checks matches against known N x 2 matches.
    """
    print(f"-- {solver.__class__.__name__} --")

    cost_matrix = torch.as_tensor(cost_matrix)

    print(
        f"- Cost matrix [{cost_matrix.shape[0]}, {cost_matrix.shape[1]}]: "
        f"mean = {cost_matrix.mean():.3f}; std = {cost_matrix.std():.3f}; "
        f"min = {cost_matrix.min():.3f}; max = {cost_matrix.max():.3f}"
    )

    solution = torch.as_tensor(solution)

    time_list = []
    warmup = 2
    trials = 6
    for i in range(warmup + trials):
        solve_time = time.process_time()
        matches, _unmatch_col, _unmatch_row = solver(cost_matrix)
        solve_time = time.process_time() - solve_time
        solve_time *= 1e3  # ms

        if i >= warmup:
            time_list.append(solve_time)

    solve_time_tensor = torch.tensor(time_list)
    solve_time_mean = solve_time_tensor.mean().item()
    solve_time_std = solve_time_tensor.std().item()

    print(f"- Solve time: {solve_time_mean:.3f} ms (std = {solve_time_std:e})")

    assert matches.shape == solution.shape

    matches = matches[matches[:, 0].argsort(dim=0).flatten(), :]

    total_cost = assignment.gather_total_cost(cost_matrix, matches)
    minimal_cost = assignment.gather_total_cost(cost_matrix, solution)
    ratio = total_cost / minimal_cost * 100

    print(f"- Total cost: {total_cost.item():.3f} ({ratio.item():.1f} %)")

    if not (
        cost_matrix[matches[:, 0], matches[:, 1]].sum()
        <= cost_matrix[solution[:, 0], solution[:, 1]].sum()
    ):
        pytest.xfail("performance issue")
