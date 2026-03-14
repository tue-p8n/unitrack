# tests/unitrack/integration/test_paper_parallel.py
"""Integration test: paper's parallel-fusion config over a deterministic clip.

Replaces a smoke-only "doesn't-crash on random tensors" assertion with
a real-fixture quality check: on the 20-frame canonical clip (4 objects,
2 occlusions) the parallel-fusion config must keep ID-switches at zero
and recover a MOTA-like score within a tolerance band. The fixture is
deterministic so the cached metric thresholds pin tracker quality
across refactors without flake.

Real-data parity benchmarks (MOTA / IDF1 against published
Mask2Former-Cityscapes numbers) live in the paper repo, where the
detector output is available; the in-tree synthetic fixture validates
end-to-end correctness without that dependency.
"""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import CDist, Cosine, MaskIoU
from unitrack.data import TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Parallel, Pipe
from unitrack.pipeline.merge import WeightedSum
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import Tracker

from .fixtures import make_canonical_clip
from .metrics import compute_metrics


def _build_parallel_tracker() -> Tracker:
    return Tracker(
        root=Pipe(
            cost=Parallel(
                children=[Cosine("kernel"), MaskIoU("mask"), CDist("centroid")],
                merge=WeightedSum([1.0, 0.5, 0.3]),
            ),
            assoc=Associate(Jonker(threshold=1.0)),
        ),
        states={
            "kernel": State(
                schema=TensorSpec(shape=(8,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
            "mask": State(
                schema=TensorSpec(shape=(4, 4), dtype=torch.bool),
                process=Identity("mask"),
                observation=Replace("mask"),
                init=FromDetectionField("mask"),
            ),
            "centroid": State(
                schema=TensorSpec(shape=(3,), dtype=torch.float32),
                process=Identity("centroid"),
                observation=Replace("centroid"),
                init=FromDetectionField("centroid"),
            ),
            "klass": State(
                schema=TensorSpec(shape=(), dtype=torch.int64),
                process=Identity("klass"),
                observation=Replace("klass"),
                init=FromDetectionField("klass"),
            ),
            "score": State(
                schema=TensorSpec(shape=(), dtype=torch.float32),
                process=Identity("score"),
                observation=Replace("score"),
                init=FromDetectionField("score"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


def test_paper_parallel_config_perfect_tracking_on_canonical_clip():
    """The parallel-fusion config tracks the 4 canonical objects across
    20 frames (2 occlusions) without a single ID switch. The objects have
    orthogonal embeddings, so cosine cost dominates and the parallel
    fusion's mask / centroid terms add discriminative signal without
    introducing spurious matches.

    Cached metrics on the canonical fixture:
      - id_switches == 0
      - consistency_rate == 1.0
      - total_observations == 78 (4 objects * 20 frames - 2 occlusion holes)
    """
    tracker = _build_parallel_tracker()
    fixture = make_canonical_clip()

    snap = tracker.empty_snapshot()
    next_id = 1
    step_results = []
    for det, ctx in zip(fixture.detections, fixture.contexts, strict=True):
        res = tracker.step(snap, det, ctx, next_id)
        snap = res.snapshot
        next_id = res.next_id
        step_results.append(res)

    metrics = compute_metrics(step_results, fixture.ground_truth)

    # Cached thresholds:
    assert metrics.misses == 0, f"tracker dropped detections: {metrics}"
    assert metrics.total_observations == 78
    assert metrics.id_switches == 0, f"unexpected ID switches: {metrics}"
    assert metrics.consistency_rate == 1.0, f"ID consistency dropped: {metrics}"
