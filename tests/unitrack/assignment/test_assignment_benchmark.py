r"""Benchmark tests for ``unitrack.assignment``."""

from __future__ import annotations

import gc
import time
from collections import defaultdict
from datetime import UTC
from pathlib import Path

import pytest
import torch
from unitrack import assignment
from unitrack.assignment import lap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parents[2] / ".pytest_cache" / "data" / "assignment"

_CUDA_AVAILABLE = torch.cuda.is_available()

# Benchmark results keyed by [dataset][algorithm] -> dict of metrics.
BENCHMARK_RESULTS: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

DATASET_SIZE = 50  # Number of random matrices per dataset type

DATASETS = [
    "iou_8x8",
    "iou_32x32",
    "iou_64x64",
    "iou_128x128",
    "iou_256x256",
    "cdist_64x64",
    "cdist_128x128",
    "dense_64x16",  # tall: more tracks than detections
    "dense_16x64",  # wide: more detections than tracks
    "gated_64x64",
    "gated_128x128",
    "empty_0x0",
]


# Thin wrapper so the benchmark can distinguish Auction-on-CPU from
# Auction-on-GPU without changing the solver class itself.
class AuctionCUDA(assignment.Auction):
    """Auction solver benchmarked on CUDA tensors."""


# Solvers that run on CPU tensors.
CPU_SOLVERS = [
    assignment.Greedy,
    assignment.Hungarian,
    assignment.Auction,
    assignment.Jonker,
    assignment.SoftAssignment,
]

# Solvers that require CUDA.
CUDA_SOLVERS = [
    AuctionCUDA,
    lap.LAP,
]

ALL_SOLVER_NAMES = [cls.__name__ for cls in CPU_SOLVERS + CUDA_SOLVERS]

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _generate_one(name: str) -> torch.Tensor:
    """Return a single cost matrix for the given dataset type.

    Cost values use scales representative of real MOT workloads:
    - ``iou``:   ``1 - IoU`` in [0, 1], the most common tracking cost.
    - ``cdist``: Euclidean distance, typically single-digit.
    - ``gated``: IoU-scale with 80% of entries masked to ``inf``.
    - ``dense``: IoU-scale without gating.
    """
    if name == "empty_0x0":
        return torch.empty((0, 0), dtype=torch.float32)
    tag, shape = name.split("_", 1)
    rows, cols = (int(x) for x in shape.split("x"))

    if tag == "iou" or tag == "dense":
        # 1 - IoU: uniform in [0, 1]
        cost = torch.rand(rows, cols, dtype=torch.float32)
    elif tag == "cdist":
        # Euclidean distance between random 2D points
        a = torch.rand(rows, 2, dtype=torch.float32) * 10
        b = torch.rand(cols, 2, dtype=torch.float32) * 10
        cost = torch.cdist(a, b)
    elif tag == "gated":
        # IoU-scale with 80% gated to inf (sparse feasible region)
        cost = torch.rand(rows, cols, dtype=torch.float32)
        mask = torch.rand(rows, cols) > 0.2
        cost[mask] = torch.inf
    else:
        msg = f"Unknown dataset tag: {tag!r} in {name!r}"
        raise ValueError(msg)

    return cost


def generate_dataset(dataset_name: str, dataset_size: int = DATASET_SIZE) -> list[dict]:
    """Generate (or load cached) benchmark matrices with optimal solutions."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / f"{dataset_name}.pt"
    if file_path.exists():
        return torch.load(file_path, weights_only=True)

    torch.manual_seed(42)
    n = 1 if dataset_name.startswith("empty_") else dataset_size
    samples = []
    for _ in range(n):
        cost = _generate_one(dataset_name)
        opt_matches, _, _ = assignment.hungarian_assignment(cost)
        opt_cost = (
            cost[opt_matches[:, 0], opt_matches[:, 1]].sum().item()
            if opt_matches.numel() > 0
            else 0.0
        )
        samples.append(
            {"cost_matrix": cost, "opt_matches": opt_matches, "opt_cost": opt_cost}
        )

    torch.save(samples, file_path)
    return samples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _measure_peak_cuda_memory(func, *args) -> float:
    """Run *func* and return peak CUDA memory allocated in MiB."""
    torch.cuda.reset_peak_memory_stats()
    func(*args)
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


_PER_SAMPLE_TIMEOUT = 5.0  # seconds; skip sample if a single solve exceeds this


def _timed_call(solver, cost_matrix):
    """Call solver with a SIGALRM guard; returns (result, elapsed) or raises."""
    import signal

    def _handler(_signum, _frame):
        msg = "solver timed out"
        raise TimeoutError(msg)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, _PER_SAMPLE_TIMEOUT)
    try:
        t0 = time.perf_counter()
        result = solver(cost_matrix)
        elapsed = time.perf_counter() - t0
        return result, elapsed
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _benchmark_solver(solver, samples, *, on_cuda: bool):  # noqa: C901, PLR0912
    """Run *solver* over *samples*, collecting latency and accuracy."""
    warmup = 3
    trials = 10
    latencies = []
    cost_ratios = []
    peak_mem_mibs = []

    for data in samples:
        cost_matrix = data["cost_matrix"]
        if on_cuda:
            cost_matrix = cost_matrix.cuda()
        opt_cost = data["opt_cost"]

        # Warmup (with timeout guard — skip this sample if too slow).
        timed_out = False
        for _ in range(warmup):
            try:
                _timed_call(solver, cost_matrix)
            except TimeoutError:
                timed_out = True
                break
        if timed_out:
            continue
        if on_cuda:
            torch.cuda.synchronize()

        # Timed trials
        trial_times = []
        for _ in range(trials):
            if on_cuda:
                torch.cuda.synchronize()
            gc.disable()
            t0 = time.perf_counter()
            matches, _, _ = solver(cost_matrix)
            if on_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            gc.enable()
            elapsed = (t1 - t0) * 1e3
            trial_times.append(elapsed)
            if elapsed > _PER_SAMPLE_TIMEOUT * 1e3:
                break

        latencies.append(torch.tensor(trial_times).median().item())

        # Peak CUDA memory (one extra call).
        if on_cuda:
            peak = _measure_peak_cuda_memory(solver, cost_matrix)
            peak_mem_mibs.append(peak)

        # Cost ratio.
        if matches.numel() > 0:
            total_cost = cost_matrix[matches[:, 0], matches[:, 1]].sum().item()
        else:
            total_cost = 0.0
        if opt_cost > 0:
            cost_ratios.append(total_cost / opt_cost)
        else:
            cost_ratios.append(1.0 if total_cost == 0 else float("inf"))

    if not latencies:
        return None  # all samples timed out

    lat = torch.tensor(latencies)
    cost_r = torch.tensor(cost_ratios)
    mem = torch.tensor(peak_mem_mibs) if peak_mem_mibs else None

    return {
        "lat_mean": lat.mean().item(),
        "lat_std": lat.std().item() if len(lat) > 1 else 0.0,
        "lat_median": lat.median().item(),
        "cost_mean": cost_r.mean().item(),
        "cost_std": cost_r.std().item() if len(cost_r) > 1 else 0.0,
        "mem_peak_mean": mem.mean().item() if mem is not None else None,
        "mem_peak_std": mem.std().item() if mem is not None and len(mem) > 1 else None,
    }


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


def _make_solver(cls):
    if issubclass(cls, assignment.Auction):
        # bid_size ~ cost_scale / N is a reasonable heuristic.
        # For IoU costs in [0,1] and N~100, bid_size=0.001 works well.
        return cls(bid_size=0.001)
    if cls == assignment.SoftAssignment:
        return cls(epsilon=0.05, num_iter=100)
    return cls()


_CUDA_MARK = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")


@pytest.mark.timeout(120)
@pytest.mark.parametrize("dataset_name", DATASETS)
@pytest.mark.parametrize(
    "solver_cls",
    [
        *CPU_SOLVERS,
        pytest.param(AuctionCUDA, marks=_CUDA_MARK),
        pytest.param(lap.LAP, marks=_CUDA_MARK),
    ],
    ids=lambda cls: cls.__name__,
)
def test_assignment_benchmark(dataset_name: str, solver_cls: type):
    samples = generate_dataset(dataset_name, dataset_size=DATASET_SIZE)
    solver = _make_solver(solver_cls)
    on_cuda = solver_cls in CUDA_SOLVERS

    results = _benchmark_solver(solver, samples, on_cuda=on_cuda)

    if results is None:
        pytest.skip(f"{solver_cls.__name__} timed out on all samples")

    BENCHMARK_RESULTS[dataset_name][solver_cls.__name__] = results

    # Correctness gate for exact solvers only.
    # Heuristic solvers (Greedy, Auction, SoftAssignment, AuctionCUDA) may
    # produce suboptimal results, especially on rectangular or degenerate
    # matrices.
    _EXACT_SOLVERS = {assignment.Hungarian, assignment.Jonker, lap.LAP}  # noqa: N806
    if not dataset_name.startswith("empty_") and solver_cls in _EXACT_SOLVERS:
        assert results["cost_mean"] <= 1.01, (
            f"{solver_cls.__name__} solution >1% worse than optimal "
            f"(ratio={results['cost_mean']:.3f})"
        )


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).parent.parents[2] / "assets" / "benchmarks"


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_json(algos: list[str]) -> None:
    """Write raw benchmark numbers to a JSON file for regression comparison."""
    import json
    from datetime import datetime

    out: dict = {
        "meta": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "cuda_available": _CUDA_AVAILABLE,
            "torch_version": torch.__version__,
            "gpu": (torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None),
            "dataset_size": DATASET_SIZE,
        },
        "results": {},
    }

    for ds in DATASETS:
        res = BENCHMARK_RESULTS.get(ds, {})
        if not res:
            continue
        out["results"][ds] = {}
        for a in algos:
            if a in res:
                out["results"][ds][a] = res[a]

    path = _RESULTS_DIR / "assignment_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[benchmark] JSON results written to {path}")


def _export_markdown(algos: list[str]) -> None:  # noqa: C901
    """Write a Markdown table suitable for inclusion in documentation."""

    def _fmt_latency(r: dict) -> str:
        return f"{r['lat_mean']:.3f} \u00b1 {r['lat_std']:.3f}"

    def _fmt_cost(r: dict) -> str:
        c = r["cost_mean"]
        if abs(c - 1.0) < 1e-4:
            return "optimal"
        return f"+{(c - 1) * 100:.1f}%"

    def _fmt_mem(r: dict) -> str:
        m = r.get("mem_peak_mean")
        if m is None:
            return "\u2014"
        return f"{m:.1f}"

    lines: list[str] = []
    lines.append("# Assignment Algorithm Benchmark\n")

    # --- Latency table ---
    lines.append("## Latency (ms)\n")
    hdr = "| Dataset |" + " | ".join(algos) + " |"
    sep = "|" + "|".join(["---"] * (1 + len(algos))) + "|"
    lines.append(hdr)
    lines.append(sep)
    for ds in DATASETS:
        res = BENCHMARK_RESULTS.get(ds, {})
        if not res:
            continue
        cells = [f" {ds} "]
        for a in algos:
            cells.append(f" {_fmt_latency(res[a])} " if a in res else " \u2014 ")  # noqa: PERF401
        lines.append("|" + "|".join(cells) + "|")

    # --- Cost quality table ---
    lines.append("\n## Solution Quality (cost ratio vs. optimal)\n")
    lines.append(hdr)
    lines.append(sep)
    for ds in DATASETS:
        res = BENCHMARK_RESULTS.get(ds, {})
        if not res:
            continue
        cells = [f" {ds} "]
        for a in algos:
            cells.append(f" {_fmt_cost(res[a])} " if a in res else " \u2014 ")
        lines.append("|" + "|".join(cells) + "|")

    # --- GPU memory table (only if any data) ---
    has_mem = any(
        BENCHMARK_RESULTS.get(ds, {}).get(a, {}).get("mem_peak_mean") is not None
        for ds in DATASETS
        for a in algos
    )
    if has_mem:
        lines.append("\n## Peak GPU Memory (MiB)\n")
        lines.append(hdr)
        lines.append(sep)
        for ds in DATASETS:
            res = BENCHMARK_RESULTS.get(ds, {})
            if not res:
                continue
            cells = [f" {ds} "]
            for a in algos:
                cells.append(f" {_fmt_mem(res[a])} " if a in res else " \u2014 ")
            lines.append("|" + "|".join(cells) + "|")

    path = _RESULTS_DIR / "assignment_benchmark.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[benchmark] Markdown results written to {path}")


# ---------------------------------------------------------------------------
# Summary table (printed + exported after all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def benchmark_table():  # noqa: C901, PLR0912
    """Print a comparison table and export results once all tests complete."""
    yield

    if not BENCHMARK_RESULTS:
        return

    algos = [
        name
        for name in ALL_SOLVER_NAMES
        if any(name in BENCHMARK_RESULTS[ds] for ds in DATASETS)
    ]
    n_algo = len(algos)
    col_w = 22
    hdr_w = 18 + (col_w + 3) * n_algo

    print("\n\n" + "=" * hdr_w)
    print(f"{'ASSIGNMENT ALGORITHM BENCHMARK':^{hdr_w}}")
    print("=" * hdr_w)

    # Header row
    hdr = f"{'Dataset':<18}"
    for a in algos:
        hdr += f" | {a:^{col_w}}"
    print(hdr)
    print("-" * hdr_w)

    for ds in DATASETS:
        res = BENCHMARK_RESULTS.get(ds, {})
        if not res:
            continue

        # Row 1: latency
        row = f"{ds:<18}"
        for a in algos:
            if a in res:
                m, s = res[a]["lat_mean"], res[a]["lat_std"]
                row += f" | {m:>8.3f} \u00b1 {s:<7.3f} ms"
            else:
                row += f" | {'\u2014':^{col_w}}"
        print(row)

        # Row 2: cost ratio
        row = f"{'':18}"
        for a in algos:
            if a in res:
                c = res[a]["cost_mean"]
                if abs(c - 1.0) < 1e-4:  # noqa: SIM108
                    val = "optimal"
                else:
                    val = f"+{(c - 1) * 100:.1f}%"
                row += f" | {'cost: ' + val:^{col_w}}"
            else:
                row += f" | {'':^{col_w}}"
        print(row)

        # Row 3: GPU memory (only if any solver reported it)
        has_mem = any(
            a in res and res[a].get("mem_peak_mean") is not None for a in algos
        )
        if has_mem:
            row = f"{'':18}"
            for a in algos:
                if a in res and res[a].get("mem_peak_mean") is not None:
                    m = res[a]["mem_peak_mean"]
                    row += f" | {'mem: ' + f'{m:.1f} MiB':^{col_w}}"
                else:
                    row += f" | {'':^{col_w}}"
            print(row)

        print("-" * hdr_w)

    # Export to files
    _export_json(algos)
    _export_markdown(algos)
