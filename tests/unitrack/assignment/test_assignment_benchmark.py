r"""Benchmark tests for ``unitrack.assignment``."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from unitrack import assignment

# Directory to save/load deterministic test matrices (set to None to disable)
DATA_DIR = Path(__file__).parent.parents[2] / ".pytest_cache"
assert DATA_DIR.is_dir(), DATA_DIR
DATA_DIR = DATA_DIR / "data" / "assignment"

# Global dictionary to accumulate benchmark results for the final table
BENCHMARK_RESULTS = defaultdict(lambda: defaultdict(dict))

DATASET_SIZE = 100  # Number of random matrices to generate per dataset type

DATASETS = [
    "dense_50x50",  # Standard balanced problem
    "tall_100x20",  # More tracks than detections
    "wide_20x100",  # More detections than tracks
    "gated_100x100",  # Sparse matrix (80% of costs are inf)
    "empty_0x0",  # Extreme edge case
    "empty_10x0",  # Extreme edge case
]


def generate_dataset_if_missing(  # noqa: C901, PLR0912
    dataset_name: str, dataset_size: int = DATASET_SIZE
) -> list[dict]:
    """Generates and saves standard cost matrices and their optimal solutions."""
    if DATA_DIR is not None:
        file_path = DATA_DIR / f"{dataset_name}.pt"
        if file_path.exists():
            return torch.load(file_path, weights_only=True)
    else:
        file_path = None

    print(f"\n[Setup] Generating missing test matrices for: {dataset_name} ...")

    torch.manual_seed(1958)

    # For nan/zero/empty edge cases, we only need 1 sample
    num_samples = 1 if dataset_name.startswith(("empty_", "nan_")) else dataset_size
    samples = []

    for _ in range(num_samples):
        # 1. Generate specific matrix types
        if dataset_name == "dense_50x50":
            cost_matrix = torch.rand(50, 50, dtype=torch.float32) * 100
        elif dataset_name == "tall_100x20":
            cost_matrix = torch.rand(100, 20, dtype=torch.float32) * 100
        elif dataset_name == "wide_20x100":
            cost_matrix = torch.rand(20, 100, dtype=torch.float32) * 100
        elif dataset_name == "gated_100x100":
            cost_matrix = torch.rand(100, 100, dtype=torch.float32) * 100
            # Simulate IoU gating by setting 80% of the matrix to infinity
            mask = torch.rand(100, 100) > 0.2
            cost_matrix[mask] = torch.inf
        elif dataset_name == "empty_0x0":
            cost_matrix = torch.empty((0, 0), dtype=torch.float32)
        elif dataset_name == "empty_10x0":
            cost_matrix = torch.empty((10, 0), dtype=torch.float32)
        else:
            msg = f"Unknown dataset: {dataset_name}"
            raise ValueError(msg)

        # 2. Calculate the global optimal solution to save as ground truth
        # We use Jonker as the reliable mathematical baseline, with a high threshold
        opt_matches, _, _ = assignment.jonker_volgenant_assignment(
            cost_matrix, threshold=1e5
        )

        if opt_matches.numel() > 0:
            opt_cost = cost_matrix[opt_matches[:, 0], opt_matches[:, 1]].sum().item()
        else:
            opt_cost = 0.0

        samples.append(
            {
                "cost_matrix": cost_matrix,
                "opt_matches": opt_matches,
                "opt_cost": opt_cost,
            }
        )

    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, file_path)
    return samples


@pytest.fixture(scope="module", autouse=True)
def benchmark_table():
    """Module-scoped fixture that prints the comparison table after all tests finish."""
    yield  # Let all tests run first

    print("\n\n" + "=" * 110)
    print(f"{'ASSIGNMENT ALGORITHM BENCHMARK (Mean ± Std (Median))':^110}")
    print("=" * 110)
    print(
        f"{'Dataset':<15} | {'Metric':<9} | {'Greedy':<18} | {'Hungarian':<18} | {'Auction':<18} | {'Jonker':<18}"  # noqa: E501
    )
    print("-" * 110)

    algos = ["Greedy", "Hungarian", "Auction", "Jonker"]

    for ds in DATASETS:
        res = BENCHMARK_RESULTS[ds]

        # Row 1: Cost/Accuracy
        cost_str = f"{ds:<15} | {'Cost +%':<9} | "
        for algo in algos:
            if algo in res:
                c_mean = res[algo]["cost_mean"]
                c_std = res[algo]["cost_std"]
                c_med = res[algo]["cost_median"]

                if ds.startswith("empty") or (
                    abs(c_mean - 1.0) < 1e-4 and c_std < 1e-4
                ):
                    val = "Optimal"
                else:
                    m_pct, s_pct, med_pct = (
                        (c_mean - 1.0) * 100,
                        c_std * 100,
                        (c_med - 1.0) * 100,
                    )
                    val = f"+{m_pct:.1f}±{s_pct:.1f} ({med_pct:.1f})"
                cost_str += f"{val:<18} | "
            else:
                cost_str += f"{'N/A':<18} | "
        print(cost_str.rstrip(" | "))  # noqa: B005

        # Row 2: Latency
        lat_str = f"{'':<15} | {'Time ms':<9} | "
        for algo in algos:
            if algo in res:
                l_mean, l_std, l_med = (
                    res[algo]["lat_mean"],
                    res[algo]["lat_std"],
                    res[algo]["lat_median"],
                )
                val = f"{l_mean:.2f}±{l_std:.2f} ({l_med:.2f})"
                lat_str += f"{val:<18} | "
            else:
                lat_str += f"{'N/A':<18} | "
        print(lat_str.rstrip(" | "))  # noqa: B005
        print("-" * 110)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("dataset_name", DATASETS)
@pytest.mark.parametrize(
    "solver_cls",
    [
        assignment.Greedy,
        assignment.Hungarian,
        assignment.Auction,
        assignment.Jonker,
    ],
    ids=lambda cls: cls.__name__,
)
def test_assignment_benchmark(dataset_name: str, solver_cls: type):
    samples = generate_dataset_if_missing(dataset_name, dataset_size=DATASET_SIZE)

    # Initialize solver (Auction needs a slightly tuned bid_size for this domain)
    if solver_cls == assignment.Auction:
        solver = solver_cls(bid_size=0.01)
    else:
        solver = solver_cls()

    algo_name = solver_cls.__name__

    latencies = []
    cost_ratios = []

    for data in samples:
        cost_matrix = data["cost_matrix"]
        opt_cost = data["opt_cost"]

        # Assert matrix mutation protection
        original_matrix = cost_matrix.clone()

        # --- Benchmark Loop ---
        warmup_trials = 3
        timed_trials = 10
        total_time = 0.0

        matches = None
        for i in range(warmup_trials + timed_trials):
            start_t = time.perf_counter()
            matches, unmatch_rows, unmatch_cols = solver(cost_matrix)
            end_t = time.perf_counter()

            if i >= warmup_trials:
                total_time += end_t - start_t

        avg_time_ms = (total_time / timed_trials) * 1000.0
        latencies.append(avg_time_ms)

        # Ensure the solver didn't modify the user's input tensor
        assert torch.equal(cost_matrix, original_matrix), (
            f"{algo_name} mutated the input tensor in-place!"
        )

        # --- Accuracy / Verification ---
        # Validate invariants
        assert (
            matches.shape[1] == 2  # ty:ignore[unresolved-attribute]
            if matches.numel() > 0  # ty:ignore[unresolved-attribute]
            else True
        )
        assert (
            not any(
                matches[:, 0] < 0  # ty:ignore[not-subscriptable]
            )
            if matches.numel() > 0  # ty:ignore[unresolved-attribute]
            else True
        )

        # Calculate cost
        if matches.numel() > 0:  # ty:ignore[unresolved-attribute]
            total_cost = (
                cost_matrix[
                    matches[:, 0],  # ty:ignore[not-subscriptable]
                    matches[:, 1],  # ty:ignore[not-subscriptable]
                ]
                .sum()
                .item()
            )
        else:
            total_cost = 0.0

        if opt_cost > 0:
            cost_ratios.append(total_cost / opt_cost)
        else:
            cost_ratios.append(1.0 if total_cost == 0 else float("inf"))

    # --- Calculate Average Statistics ---
    lat_tensor = torch.tensor(latencies)
    cost_tensor = torch.tensor(cost_ratios)

    # Store results for the table fixture
    BENCHMARK_RESULTS[dataset_name][algo_name] = {
        "cost_mean": cost_tensor.mean().item(),
        "cost_std": cost_tensor.std().item() if len(cost_tensor) > 1 else 0.0,
        "cost_median": cost_tensor.median().item(),
        "lat_mean": lat_tensor.mean().item(),
        "lat_std": lat_tensor.std().item() if len(lat_tensor) > 1 else 0.0,
        "lat_median": lat_tensor.median().item(),
    }

    if not dataset_name.startswith("empty_"):
        assert cost_tensor.mean().item() <= 1.5, (
            f"{algo_name} generated a solution >50% worse than optimal on average."
        )
