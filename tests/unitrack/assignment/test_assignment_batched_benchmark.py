r"""Benchmark for batched linear assignment — realistic object-tracking regime.

Matrix sizes span 8–256 detections; batch sizes (1, 4, 8) reflect single-
camera through multi-camera deployments.  Both square (equal track/detection
count) and rectangular (more detections than tracks, the common case when new
objects enter the frame) shapes are included, along with a gated variant where
~80 % of entries are masked to ``inf`` (geometric gate or class mismatch).
"""
# ruff: noqa: C901, N806, PERF401, PLR0912, PLR0915, PLW3301, RUF002

from __future__ import annotations

import gc
import os
import time
from collections import defaultdict
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest
import torch
from unitrack import assignment
from unitrack.assignment.lap import Backend, lap_assignment, lap_batch_assignment
from unitrack.assignment.lapjv import (
    lapjvs_assignment as _lapjvs_assignment,
)
from unitrack.assignment.lapjv import (
    lapjvs_batch_assignment as _lapjvs_batch_assignment,
)
from unitrack.assignment.lapjv import (
    lapjvx_assignment as _lapjvx_assignment,
)
from unitrack.assignment.lapjv import (
    lapjvx_batch_assignment as _lapjvx_batch_assignment,
)

try:
    import lap as lapx

    _LAPX_AVAILABLE = (
        hasattr(lapx, "lapjvc")
        and hasattr(lapx, "lapjvs")
        and hasattr(lapx, "lapjvx_batch")
    )
except ImportError:
    _LAPX_AVAILABLE = False


_CUDA_AVAILABLE = torch.cuda.is_available()

DATA_DIR = (
    Path(__file__).parent.parents[2] / ".pytest_cache" / "data" / "assignment_batched"
)
_RESULTS_DIR = Path(__file__).parent.parents[2] / "assets" / "benchmarks"

# (rows, cols, batch, tag)
# rows = tracks (prior objects), cols = detections (current frame).
# Rectangular configs use a 3:2 ratio — a common tracking scenario where
# more detections than tracks arrive (new objects entering the scene).
CONFIGS: list[tuple[int, int, int, str]] = [
    # ── Small: n ∈ {8, 16, 32} ─────────────────────────────────────────────
    (8, 8, 1, "iou"),  # single camera, tiny scene
    (8, 8, 8, "iou"),
    (8, 12, 8, "iou"),  # 8 tracks, 12 detections
    (16, 16, 1, "iou"),
    (16, 16, 8, "iou"),
    (16, 24, 8, "iou"),  # 16 tracks, 24 detections
    (32, 32, 1, "iou"),
    (32, 32, 8, "iou"),
    (32, 48, 8, "iou"),  # 32 tracks, 48 detections
    # ── Medium: n ∈ {64, 128} ───────────────────────────────────────────────
    (64, 64, 1, "iou"),
    (64, 64, 8, "iou"),
    (64, 96, 8, "iou"),  # 64 tracks, 96 detections
    (128, 128, 1, "iou"),
    (128, 128, 4, "iou"),
    (128, 192, 4, "iou"),  # 128 tracks, 192 detections
    # ── Large: n = 256 ──────────────────────────────────────────────────────
    (256, 256, 1, "iou"),
    (256, 256, 4, "iou"),
    # ── Gated (sparse): ~80 % of entries masked to inf ─────────────────────
    (32, 32, 4, "gated"),
    (64, 64, 4, "gated"),
    (128, 128, 4, "gated"),
]


def _config_id(cfg: tuple[int, int, int, str]) -> str:
    rows, cols, batch, tag = cfg
    return f"{tag}_{rows}x{cols}_B{batch}"


# Benchmark results keyed by [config_id][algorithm] -> dict of metrics.
BENCHMARK_RESULTS: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _generate_one(tag: str, rows: int, cols: int) -> torch.Tensor:
    if tag == "iou":
        return torch.rand(rows, cols, dtype=torch.float32)
    if tag == "gated":
        # Sparse feasible region: ~20% finite entries, plus a guaranteed
        # feasible permutation so a perfect matching always exists.
        cost = torch.rand(rows, cols, dtype=torch.float32)
        mask = torch.rand(rows, cols) > 0.2
        cost[mask] = torch.inf
        n = min(rows, cols)
        perm = torch.randperm(n)
        for r in range(n):
            cost[r, perm[r]] = torch.rand(1).item()
        return cost
    msg = f"unknown shape tag: {tag!r}"
    raise ValueError(msg)


def _generate_dataset(cfg: tuple[int, int, int, str]) -> list[dict]:
    rows, cols, batch, tag = cfg
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{_config_id(cfg)}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)

    torch.manual_seed(42)
    samples = []
    for _ in range(batch):
        cost = _generate_one(tag, rows, cols)
        opt_matches, _, _ = assignment.hungarian_assignment(cost)
        opt_cost = (
            cost[opt_matches[:, 0], opt_matches[:, 1]].sum().item()
            if opt_matches.numel() > 0
            else 0.0
        )
        samples.append({"cost_matrix": cost, "opt_cost": opt_cost})

    torch.save(samples, path)
    return samples


# ---------------------------------------------------------------------------
# Solvers under test
# ---------------------------------------------------------------------------


def _solver_batched(costs_cuda: list[torch.Tensor]):
    return lap_batch_assignment(costs_cuda)


def _make_loop_solver(backend: int):
    def _run(costs_cuda: list[torch.Tensor]):
        return [lap_assignment(c, backend=backend) for c in costs_cuda]

    _run.__name__ = f"lap_loop_backend{backend}"
    return _run


def _solver_scipy_cpu(costs_cpu: list[torch.Tensor]):
    return [assignment.hungarian_assignment(c) for c in costs_cpu]


def _lapx_to_matches(rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int):
    """Convert lapjvc/lapjvs SciPy-style output into the
    (matches, unmatched_rows, unmatched_cols) triple."""
    matches = np.column_stack((rows, cols))
    matches = matches[(matches[:, 0] < n_rows) & (matches[:, 1] < n_cols)]
    matches_t = torch.from_numpy(matches).long()
    rows_idx = torch.arange(n_rows, dtype=torch.long)
    cols_idx = torch.arange(n_cols, dtype=torch.long)
    unmatched_rows = rows_idx[~torch.isin(rows_idx, matches_t[:, 0])]
    unmatched_cols = cols_idx[~torch.isin(cols_idx, matches_t[:, 1])]
    return matches_t, unmatched_rows, unmatched_cols


def _solver_lapx_lapjvc(costs_cpu: list[torch.Tensor]):
    out = []
    for c in costs_cpu:
        cm = c.numpy().astype(np.float64)
        cm = np.where(np.isfinite(cm), cm, np.inf)
        _, rows, cols = lapx.lapjvc(cm, return_cost=True)
        out.append(_lapx_to_matches(rows, cols, c.size(0), c.size(1)))
    return out


def _solver_lapx_lapjvs(costs_cpu: list[torch.Tensor]):
    out = []
    for c in costs_cpu:
        cm = c.numpy().astype(np.float64)
        cm = np.where(np.isfinite(cm), cm, np.inf)
        _, rows, cols = lapx.lapjvs(cm, return_cost=True, jvx_like=True)
        out.append(_lapx_to_matches(rows, cols, c.size(0), c.size(1)))
    return out


def _solver_lapx_lapjvx_batch(costs_cpu: list[torch.Tensor]):
    """Stack into (B, K, K) and call lapjvx_batch with thread pool."""
    n_rows = [c.size(0) for c in costs_cpu]
    n_cols = [c.size(1) for c in costs_cpu]
    K = max(max(n_rows), max(n_cols))
    B = len(costs_cpu)

    # Build padded square batch with a per-batch sentinel.
    batch = np.empty((B, K, K), dtype=np.float64)
    for i, c in enumerate(costs_cpu):
        cm = c.numpy().astype(np.float64)
        cm = np.where(np.isfinite(cm), cm, np.inf)
        finite = np.isfinite(cm)
        sentinel = (cm[finite].max() + 1.0) * (K + 1.0) if finite.any() else 1.0
        cm = np.where(finite, cm, sentinel)
        sub = np.full((K, K), sentinel, dtype=np.float64)
        sub[: c.size(0), : c.size(1)] = cm
        batch[i] = sub

    _, rows_list, cols_list = lapx.lapjvx_batch(
        batch,
        return_cost=True,
        n_threads=os.cpu_count() or 1,
    )

    out = []
    for i in range(B):
        # Restrict to original (n_rows[i], n_cols[i]) tile.
        r = rows_list[i]
        c = cols_list[i]
        # Filter pairs where row < n_rows[i] and col < n_cols[i].
        keep = (r < n_rows[i]) & (c < n_cols[i])
        out.append(_lapx_to_matches(r[keep], c[keep], n_rows[i], n_cols[i]))
    return out


def _solver_lapx_lapjvx_batch_via_gpu(costs_cuda: list[torch.Tensor]):
    """lapjvx_batch but pay the GPU↔CPU round-trip per problem."""
    cpu_costs = [c.cpu() for c in costs_cuda]
    results = _solver_lapx_lapjvx_batch(cpu_costs)
    return [
        (
            m.cuda(non_blocking=False),
            ur.cuda(non_blocking=False),
            uc.cuda(non_blocking=False),
        )
        for (m, ur, uc) in results
    ]


def _solver_unitrack_lapjvx_loop(costs_cpu: list[torch.Tensor]):
    return [_lapjvx_assignment(c) for c in costs_cpu]


def _solver_unitrack_lapjvs_loop(costs_cpu: list[torch.Tensor]):
    return [_lapjvs_assignment(c) for c in costs_cpu]


def _solver_unitrack_lapjvx_batch(costs_cpu: list[torch.Tensor]):
    return _lapjvx_batch_assignment(costs_cpu)


def _solver_unitrack_lapjvs_batch(costs_cpu: list[torch.Tensor]):
    return _lapjvs_batch_assignment(costs_cpu)


def _solver_scipy_via_gpu(costs_cuda: list[torch.Tensor]):
    """SciPy Hungarian, but pay GPU->CPU and CPU->GPU transfers per problem.

    Simulates the realistic pipeline cost when surrounding computations
    run on the GPU: cost matrices arrive on device, must be copied to
    host for SciPy, and the resulting index tensors are returned to
    device for downstream use.
    """
    results = []
    for c_cuda in costs_cuda:
        c_cpu = c_cuda.cpu()
        matches, ur, uc = assignment.hungarian_assignment(c_cpu)
        results.append(
            (
                matches.cuda(non_blocking=False),
                ur.cuda(non_blocking=False),
                uc.cuda(non_blocking=False),
            )
        )
    return results


# Algorithm name -> (callable, needs_cuda_inputs).
ALGORITHMS: dict[str, tuple[object, bool]] = {
    "lap_batch_cuda": (_solver_batched, True),
    "lap_loop_classical": (_make_loop_solver(Backend.CLASSICAL), True),
    "lap_loop_hybrid": (_make_loop_solver(Backend.HYBRID), True),
    "lap_loop_tree": (_make_loop_solver(Backend.TREE), True),
    "hungarian_loop_cpu": (_solver_scipy_cpu, False),
    "hungarian_loop_via_gpu": (_solver_scipy_via_gpu, True),
    "lapx_lapjvc_loop": (_solver_lapx_lapjvc, False),
    "lapx_lapjvs_loop": (_solver_lapx_lapjvs, False),
    "lapx_lapjvx_batch": (_solver_lapx_lapjvx_batch, False),
    "lapx_lapjvx_batch_via_gpu": (_solver_lapx_lapjvx_batch_via_gpu, True),
    "unitrack_lapjvx_loop": (_solver_unitrack_lapjvx_loop, False),
    "unitrack_lapjvs_loop": (_solver_unitrack_lapjvs_loop, False),
    "unitrack_lapjvx_batch": (_solver_unitrack_lapjvx_batch, False),
    "unitrack_lapjvs_batch": (_solver_unitrack_lapjvs_batch, False),
}

# Algorithms that must produce optimal assignments.
_EXACT = set(ALGORITHMS.keys())


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


_WARMUP = 3
_TRIALS = 10
_CFG_TIMEOUT_S = 10.0  # skip algorithm if a single trial exceeds this


def _run_algo(fn, samples, *, on_cuda: bool) -> dict | None:
    """Time ``fn`` once per trial over the whole batch.

    Latency reported is wall time for a single invocation on the full
    list of ``len(samples)`` matrices. Cost ratio is averaged across
    the batch against the SciPy baseline.
    """
    if on_cuda:
        inputs = [s["cost_matrix"].cuda() for s in samples]
    else:
        inputs = [s["cost_matrix"] for s in samples]

    # Warmup
    for _ in range(_WARMUP):
        fn(inputs)
    if on_cuda:
        torch.cuda.synchronize()

    # Timed trials
    trial_times = []
    for _ in range(_TRIALS):
        if on_cuda:
            torch.cuda.synchronize()
        gc.disable()
        t0 = time.perf_counter()
        results = fn(inputs)
        if on_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        gc.enable()
        elapsed = (t1 - t0) * 1e3
        trial_times.append(elapsed)
        if elapsed > _CFG_TIMEOUT_S * 1e3:
            return None

    # Accuracy (one extra call; results already cached from last trial)
    cost_ratios = []
    for i, (matches, _, _) in enumerate(results):
        cost_m = inputs[i]
        opt = samples[i]["opt_cost"]
        if matches.numel() > 0:
            total = cost_m[matches[:, 0], matches[:, 1]].sum().item()
        else:
            total = 0.0
        if opt > 0:
            cost_ratios.append(total / opt)
        else:
            cost_ratios.append(1.0 if total == 0 else float("inf"))

    # Peak CUDA memory: one extra call under reset.
    peak_mib = None
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
        fn(inputs)
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    lat = torch.tensor(trial_times)
    cost_r = torch.tensor(cost_ratios)
    B = len(samples)
    return {
        "batch_size": B,
        "lat_mean_ms": lat.mean().item(),
        "lat_std_ms": lat.std().item() if len(lat) > 1 else 0.0,
        "lat_median_ms": lat.median().item(),
        "lat_per_problem_us": lat.median().item() * 1e3 / B,
        "cost_mean": cost_r.mean().item(),
        "cost_std": cost_r.std().item() if len(cost_r) > 1 else 0.0,
        "mem_peak_mib": peak_mib,
    }


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


_CUDA_MARK = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")
_LAPX_MARK = pytest.mark.skipif(not _LAPX_AVAILABLE, reason="lapx required")
_LAPX_CUDA_MARK = pytest.mark.skipif(
    not (_LAPX_AVAILABLE and _CUDA_AVAILABLE),
    reason="lapx + CUDA required",
)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("config", CONFIGS, ids=_config_id)
@pytest.mark.parametrize(
    "algo_name",
    [
        "hungarian_loop_cpu",
        "unitrack_lapjvx_loop",
        "unitrack_lapjvs_loop",
        "unitrack_lapjvx_batch",
        "unitrack_lapjvs_batch",
        pytest.param("lap_batch_cuda", marks=_CUDA_MARK),
        pytest.param("lap_loop_classical", marks=_CUDA_MARK),
        pytest.param("lap_loop_hybrid", marks=_CUDA_MARK),
        pytest.param("lap_loop_tree", marks=_CUDA_MARK),
        pytest.param("hungarian_loop_via_gpu", marks=_CUDA_MARK),
        pytest.param("lapx_lapjvc_loop", marks=_LAPX_MARK),
        pytest.param("lapx_lapjvs_loop", marks=_LAPX_MARK),
        pytest.param("lapx_lapjvx_batch", marks=_LAPX_MARK),
        pytest.param("lapx_lapjvx_batch_via_gpu", marks=_LAPX_CUDA_MARK),
    ],
)
def test_batched_assignment_benchmark(config, algo_name):
    samples = _generate_dataset(config)
    fn, on_cuda = ALGORITHMS[algo_name]

    res = _run_algo(fn, samples, on_cuda=on_cuda)
    if res is None:
        pytest.skip(f"{algo_name} exceeded per-trial timeout on {_config_id(config)}")

    BENCHMARK_RESULTS[_config_id(config)][algo_name] = res

    if algo_name in _EXACT:
        assert res["cost_mean"] <= 1.01, (
            f"{algo_name} cost_mean={res['cost_mean']:.4f} > 1.01 on "
            f"{_config_id(config)}"
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_json(algos: list[str]) -> None:
    import json
    from datetime import datetime

    out: dict = {
        "meta": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "cuda_available": _CUDA_AVAILABLE,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None,
            "warmup": _WARMUP,
            "trials": _TRIALS,
        },
        "results": {},
    }
    for cfg in CONFIGS:
        cid = _config_id(cfg)
        res = BENCHMARK_RESULTS.get(cid, {})
        if not res:
            continue
        out["results"][cid] = {a: res[a] for a in algos if a in res}

    path = _RESULTS_DIR / "assignment_batched_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[batched benchmark] JSON written to {path}")


def _export_markdown(algos: list[str]) -> None:
    def _fmt_ms(r: dict) -> str:
        return f"{r['lat_mean_ms']:.3f} \u00b1 {r['lat_std_ms']:.3f}"

    def _fmt_us(r: dict) -> str:
        return f"{r['lat_per_problem_us']:.1f}"

    def _fmt_cost(r: dict) -> str:
        c = r["cost_mean"]
        if abs(c - 1.0) < 1e-4:
            return "optimal"
        return f"+{(c - 1) * 100:.2f}%"

    def _fmt_mem(r: dict) -> str:
        m = r.get("mem_peak_mib")
        return "\u2014" if m is None else f"{m:.1f}"

    lines: list[str] = ["# Batched Assignment Benchmark\n"]

    # --- Total batch latency ---
    lines.append("## Total batch latency (ms)\n")
    hdr = "| Config |" + " | ".join(algos) + " |"
    sep = "|" + "|".join(["---"] * (1 + len(algos))) + "|"
    lines.append(hdr)
    lines.append(sep)
    for cfg in CONFIGS:
        cid = _config_id(cfg)
        res = BENCHMARK_RESULTS.get(cid, {})
        if not res:
            continue
        cells = [f" {cid} "]
        for a in algos:
            cells.append(f" {_fmt_ms(res[a])} " if a in res else " \u2014 ")
        lines.append("|" + "|".join(cells) + "|")

    # --- Per-problem amortized latency ---
    lines.append("\n## Amortized per-problem latency (\u00b5s)\n")
    lines.append(hdr)
    lines.append(sep)
    for cfg in CONFIGS:
        cid = _config_id(cfg)
        res = BENCHMARK_RESULTS.get(cid, {})
        if not res:
            continue
        cells = [f" {cid} "]
        for a in algos:
            cells.append(f" {_fmt_us(res[a])} " if a in res else " \u2014 ")
        lines.append("|" + "|".join(cells) + "|")

    # --- Solution quality ---
    lines.append("\n## Solution Quality (cost ratio vs. SciPy)\n")
    lines.append(hdr)
    lines.append(sep)
    for cfg in CONFIGS:
        cid = _config_id(cfg)
        res = BENCHMARK_RESULTS.get(cid, {})
        if not res:
            continue
        cells = [f" {cid} "]
        for a in algos:
            cells.append(f" {_fmt_cost(res[a])} " if a in res else " \u2014 ")
        lines.append("|" + "|".join(cells) + "|")

    # --- Peak GPU memory ---
    has_mem = any(
        BENCHMARK_RESULTS.get(_config_id(cfg), {}).get(a, {}).get("mem_peak_mib")
        is not None
        for cfg in CONFIGS
        for a in algos
    )
    if has_mem:
        lines.append("\n## Peak GPU Memory (MiB)\n")
        lines.append(hdr)
        lines.append(sep)
        for cfg in CONFIGS:
            cid = _config_id(cfg)
            res = BENCHMARK_RESULTS.get(cid, {})
            if not res:
                continue
            cells = [f" {cid} "]
            for a in algos:
                cells.append(f" {_fmt_mem(res[a])} " if a in res else " \u2014 ")
            lines.append("|" + "|".join(cells) + "|")

    path = _RESULTS_DIR / "assignment_batched_benchmark.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[batched benchmark] Markdown written to {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


_ALGO_ORDER = [
    "unitrack_lapjvx_loop",
    "unitrack_lapjvs_loop",
    "unitrack_lapjvx_batch",
    "unitrack_lapjvs_batch",
    "lap_batch_cuda",
    "lap_loop_classical",
    "lap_loop_hybrid",
    "lap_loop_tree",
    "hungarian_loop_cpu",
    "hungarian_loop_via_gpu",
    "lapx_lapjvc_loop",
    "lapx_lapjvs_loop",
    "lapx_lapjvx_batch",
    "lapx_lapjvx_batch_via_gpu",
]


@pytest.fixture(scope="module", autouse=True)
def _summary():
    yield
    if not BENCHMARK_RESULTS:
        return

    algos = [
        a
        for a in _ALGO_ORDER
        if any(a in BENCHMARK_RESULTS[c] for c in BENCHMARK_RESULTS)
    ]

    col_w = 22
    hdr_w = 24 + (col_w + 3) * len(algos)
    print("\n\n" + "=" * hdr_w)
    print(f"{'BATCHED ASSIGNMENT BENCHMARK':^{hdr_w}}")
    print("=" * hdr_w)

    hdr = f"{'Config':<24}"
    for a in algos:
        hdr += f" | {a:^{col_w}}"
    print(hdr)
    print("-" * hdr_w)

    for cfg in CONFIGS:
        cid = _config_id(cfg)
        res = BENCHMARK_RESULTS.get(cid, {})
        if not res:
            continue

        row = f"{cid:<24}"
        for a in algos:
            if a in res:
                m = res[a]["lat_mean_ms"]
                s = res[a]["lat_std_ms"]
                row += f" | {m:>8.3f} \u00b1 {s:<7.3f} ms"
            else:
                row += f" | {'\u2014':^{col_w}}"
        print(row)

        row = f"{'':24}"
        for a in algos:
            if a in res:
                us = res[a]["lat_per_problem_us"]
                row += f" | {f'{us:.1f} \u00b5s/prob':^{col_w}}"
            else:
                row += f" | {'':^{col_w}}"
        print(row)
        print("-" * hdr_w)

    _export_json(algos)
    _export_markdown(algos)
