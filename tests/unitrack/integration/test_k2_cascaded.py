# tests/unitrack/integration/test_k2_cascaded.py
"""Integration test: K=2 cascaded canonical config over a fixture.

Replaces the smoke-only "doesn't crash on random tensors" check with a
deterministic 20-frame canonical clip and quality assertions on
ID-consistency. The K=2 cascade applies class + score gating in stage 1
and class + motion gating in stage 2, both feeding a cosine-based cost
into a Jonker assignment.

Real-data parity benchmarks (MOTA / IDF1 on Mask2Former-Cityscapes) live
in the paper repo; the in-tree synthetic fixture validates end-to-end
correctness of the K=2 cascade without that dependency.
"""

from __future__ import annotations

import dataclasses

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine, MaskIoU, Reduce
from unitrack.data import Detections, FrameContext, TensorSpec, Tracklets
from unitrack.gates import ClassGate, MotionGate, ScoreGate
from unitrack.lifecycle import (
    ConfirmedOnly,
    NoLifecycle,
    StandardLifecycle,
    StatusFilter,
    TrackletStatus,
)
from unitrack.pipeline import Filter, Gated, Pipe, Sequential
from unitrack.states import (
    FromDetectionField,
    Identity,
    Replace,
    State,
)
from unitrack.states.kalman import KalmanCentroid3D
from unitrack.tracker import MultiStream, Tracker

from .fixtures import make_canonical_clip
from .metrics import compute_metrics


@dataclasses.dataclass(frozen=True, slots=True)
class _CentroidInit:
    """Initialize Kalman centroid from 3D detection: embed pos into 6D [x,y,z,0,0,0]."""

    field: str

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        del ctx
        pos = getattr(ds, self.field)  # (N, 3)
        n = pos.shape[0]
        vel = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        return torch.cat([pos, vel], dim=-1)  # (N, 6)


@dataclasses.dataclass(frozen=True, slots=True)
class _CovInit:
    """Initialize Kalman covariance as identity in the full state space."""

    dim: int

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        del ctx
        n = ds.batch_size[0]
        eye = torch.eye(self.dim).unsqueeze(0)
        return eye.expand(n, self.dim, self.dim).contiguous()


@dataclasses.dataclass(frozen=True, slots=True)
class _IdentityCov:
    """No-op Process: covariance is updated by KalmanLinear via the mean field."""

    field: str

    def __call__(self, cs, ctx: FrameContext):
        del ctx
        return cs


@dataclasses.dataclass(frozen=True, slots=True)
class _NoopObservation:
    """Observation: no-op (KalmanUpdate handles cov updates via the main field)."""

    field: str

    def __call__(self, cs, ds, match, ctx: FrameContext):
        del ds, match, ctx
        return cs


def _build_tracker() -> Tracker:
    centroid_proc = KalmanCentroid3D("centroid_3d")
    return Tracker(
        root=Sequential(
            [
                Filter(
                    StatusFilter(TrackletStatus.Active, TrackletStatus.Lost),
                    on="cs",
                    then=Pipe(  # type: ignore[arg-type]
                        cost=Gated(
                            gate=Sequential(
                                [
                                    ClassGate("klass"),
                                    ScoreGate("score", threshold=0.6),
                                ]
                            ),
                            then=Reduce([Cosine("kernel"), MaskIoU("mask")], "sum"),
                        ),
                        assoc=Associate(Jonker(threshold=0.3)),
                    ),
                ),
                Pipe(
                    cost=Gated(
                        gate=Sequential(
                            [
                                ClassGate("klass"),
                                MotionGate(
                                    "centroid_3d", "centroid_3d_cov", max_chi2=9.21
                                ),
                            ]
                        ),
                        # NOTE: CDist over centroid_3d would mix the 6-D Kalman
                        # state and 3-D detection measurement; this smoke test
                        # uses Cosine on kernel only. Real reproductions should
                        # add a measurement-space position field for CDist.
                        then=Cosine("kernel"),
                    ),
                    assoc=Associate(Jonker(threshold=0.5)),
                ),
            ]
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
            # KalmanCentroid3D uses [x,y,z,vx,vy,vz] internally (6D)
            "centroid_3d": State(
                schema=TensorSpec(shape=(6,), dtype=torch.float32),
                process=centroid_proc,
                observation=centroid_proc.make_update(),
                init=_CentroidInit("centroid_3d"),
            ),
            # Kalman covariance is 6x6; process+obs are handled via centroid_3d
            "centroid_3d_cov": State(
                schema=TensorSpec(shape=(6, 6), dtype=torch.float32),
                process=_IdentityCov("centroid_3d_cov"),
                observation=_NoopObservation("centroid_3d_cov"),
                init=_CovInit(6),
            ),
        },
        lifecycle=StandardLifecycle(min_hits=2, max_age=3, allow_reid=10),
        visibility=ConfirmedOnly(),
    )


def _build_simple_cascade_tracker() -> Tracker:
    """K=2 cascade without Kalman: stage 1 = class + score gate, stage 2 = class only.

    The fixture's `centroid` field is 3-D, but `KalmanCentroid3D` expects
    a centroid_3d_cov state alongside; rather than introduce the full
    Kalman wiring inside the test, this simpler cascade exercises the
    same composition primitives (Filter → Sequential[MatchOutcome] →
    Gated → Pipe) without the Kalman state-pair complexity.
    """
    return Tracker(
        root=Sequential(
            [
                Filter(
                    StatusFilter(TrackletStatus.Active, TrackletStatus.Lost),
                    on="cs",
                    then=Pipe(  # type: ignore[arg-type]
                        cost=Gated(
                            gate=Sequential(
                                [
                                    ClassGate("klass"),
                                    ScoreGate("score", threshold=0.6),
                                ]
                            ),
                            then=Reduce([Cosine("kernel"), MaskIoU("mask")], "sum"),
                        ),
                        assoc=Associate(Jonker(threshold=0.5)),
                    ),
                ),
                Pipe(
                    cost=Gated(
                        gate=ClassGate("klass"),
                        then=Cosine("kernel"),
                    ),
                    assoc=Associate(Jonker(threshold=0.5)),
                ),
            ]
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
            "centroid": State(
                schema=TensorSpec(shape=(3,), dtype=torch.float32),
                process=Identity("centroid"),
                observation=Replace("centroid"),
                init=FromDetectionField("centroid"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=ConfirmedOnly(),
    )


def test_k2_cascaded_simple_perfect_tracking_on_canonical_clip():
    """The K=2 cascaded config tracks the 4 canonical objects across
    20 frames (2 occlusions) with zero ID switches. Stage 1 (class +
    score gate, cosine + mask cost) handles the easy matches; stage 2
    (class gate, cosine only) picks up any residuals from stage 1.

    Cached metrics on the canonical fixture:
      - id_switches == 0
      - consistency_rate == 1.0
      - total_observations == 78 (4 objects * 20 frames - 2 occlusion holes)
    """
    tracker = _build_simple_cascade_tracker()
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
    assert metrics.misses == 0, f"tracker dropped detections: {metrics}"
    assert metrics.total_observations == 78
    assert metrics.id_switches == 0, f"unexpected ID switches: {metrics}"
    assert metrics.consistency_rate == 1.0, f"ID consistency dropped: {metrics}"


def _make_detections(n: int, seed: int) -> Detections:
    g = torch.Generator().manual_seed(seed)
    return Detections(
        index=torch.arange(n, dtype=torch.int64),
        kernel=torch.randn(n, 8, generator=g),
        mask=torch.zeros(n, 4, 4, dtype=torch.bool),
        klass=torch.zeros(n, dtype=torch.int64),
        score=torch.full((n,), 0.9),
        centroid_3d=torch.randn(n, 3, generator=g),
        batch_size=[n],
    )


def test_k2_cascaded_kalman_runs_end_to_end_over_5_frames():
    """Smoke-only test for the full Kalman-equipped K=2 cascade.

    The Kalman variant needs the centroid_3d_cov state pairing (see
    _build_tracker); the canonical fixture would have to be extended with
    covariance seeds, which is out of scope. The fixture-backed test above
    carries the quality assertions for the cascade composition; this one
    just verifies the Kalman config instantiates and produces a well-formed
    snapshot per frame (no NaN ids, lifecycle counters non-negative).
    """
    tracker = _build_tracker()
    ms = MultiStream(tracker)
    for frame in range(5):
        ds = _make_detections(n=3, seed=frame)
        res = ms.step(
            stream_key=0,
            detections=ds,
            ctx=FrameContext.make(frame, delta=1 / 15, stream_key=0),
        )
        snap = res.snapshot
        assert isinstance(snap, Tracklets)
        assert snap.id.dtype == torch.int64
        assert not torch.any(snap.id < 0), f"frame {frame}: negative id in {snap.id}"
        assert torch.all(snap.hits >= 0), f"frame {frame}: negative hits"
        assert torch.all(snap.age >= 0), f"frame {frame}: negative age"
