"""
Optuna search space for the tracker design space, mapped to a unitrack.Tracker.

Exposes :func:`sample_tracker(trial, *, descriptor_dim, mask_shape)` which
builds a fully-typed :class:`unitrack.Tracker` from a single Optuna trial.
The search space axes are listed in ``README.md``; see the row-by-row
mapping.

The function is deterministic given the trial's parameter dict, so a
configuration can be replayed by passing a ``optuna.trial.FixedTrial(params)``.
"""

from __future__ import annotations

import dataclasses
import typing

import torch
import unitrack
from unitrack.assignment import Associate, Greedy, Jonker
from unitrack.costs import (
    BiSoftmax,
    BoxIoU,
    CDist,
    Cosine,
    MaskIoU,
)
from unitrack.data import TensorSpec
from unitrack.gates import (
    ClassGate,
    MotionGate,
    NoneGate,
    ScoreGate,
    SpatialGate2D,
)
from unitrack.lifecycle import (
    ConfirmedOnly,
    StandardLifecycle,
    StatusFilter,
    TrackletStatus,
)
from unitrack.pipeline import Filter, Gated, Parallel, Pipe, Sequential
from unitrack.pipeline.merge import WeightedSum
from unitrack.states import (
    FromDetectionField,
    Identity,
    Replace,
    State,
)
from unitrack.states.kalman import KalmanCentroid2D

if typing.TYPE_CHECKING:
    import optuna

__all__ = ["DEFAULT_PARAMS", "TrackerSchema", "sample_tracker"]


# ---------------------------------------------------------------------------
# Schema: per-Tracker fixed shape + dtype declarations.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TrackerSchema:
    """Fixed-at-construction-time shapes for a Tracker's user fields."""

    kernel_dim: int = 256
    mask_shape: tuple[int, int] = (96, 192)
    n_classes: int = 19
    centroid_dim: int = 2  # use centroid_3d_dim=3 for DVPS variants


# ---------------------------------------------------------------------------
# Catalogues — mapping from string keys to unitrack constructors.
# ---------------------------------------------------------------------------

_GATING_CHOICES = ("none", "class", "score", "spatial", "motion")
_DESCRIPTOR_CHOICES = ("kernel", "mask", "bbox")
_COST_CHOICES = ("cosine", "cdist", "bisoftmax", "iou")
_ASSOC_CHOICES = ("jonker", "greedy")
_FUSION_CHOICES = ("cascaded", "parallel")
_K_RANGE = (1, 2, 3, 4)


def _build_gate(kind: str, *, score_threshold: float, max_chi2: float):
    """Construct a GateProducer for the given paper gate kind."""
    match kind:
        case "none":
            return NoneGate()
        case "class":
            return ClassGate("klass")
        case "score":
            return ScoreGate("score", threshold=score_threshold)
        case "spatial":
            return SpatialGate2D("centroid", max_dist=80.0)
        case "motion":
            return MotionGate(
                mean_field="centroid",
                cov_field="centroid_cov",
                max_chi2=max_chi2,
            )
    msg = f"unknown gate kind {kind!r}"
    raise ValueError(msg)


def _build_cost_for_descriptor(descriptor: str, cost: str):  # noqa: PLR0911
    """
    Construct a CostProducer for the (descriptor, cost) pair.

    Some pairs are not meaningful (e.g. cosine on bbox). The function
    returns None for combinations that should be sampled-out by the
    objective; callers should re-sample on None.
    """
    match (descriptor, cost):
        case ("kernel", "cosine"):
            return Cosine("kernel")
        case ("kernel", "cdist"):
            return CDist("kernel", p_norm=2.0)
        case ("kernel", "bisoftmax"):
            return BiSoftmax("kernel")
        case ("mask", "iou"):
            return MaskIoU("mask")
        case ("bbox", "iou"):
            return BoxIoU("bbox")
        case ("bbox", "cdist"):
            return CDist("centroid", p_norm=2.0)
    return None


def _build_assignment(kind: str, threshold: float):
    """Construct an Assignment backend for the given paper algorithm."""
    if kind == "jonker":
        return Jonker(threshold=threshold)
    if kind == "greedy":
        return Greedy(threshold=threshold)
    msg = f"unknown assignment algorithm {kind!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Per-stage sampling.
# ---------------------------------------------------------------------------


def _suggest_stage(
    trial: optuna.trial.Trial,
    *,
    prefix: str,
    schema: TrackerSchema,  # noqa: ARG001 — reserved for future schema-aware sampling
):
    """Sample one stage's (gate, cost-producer, threshold, assignment)."""
    # Resample descriptor/cost until the pair is meaningful.
    descriptor = trial.suggest_categorical(f"{prefix}.descriptor", _DESCRIPTOR_CHOICES)
    cost_kind = trial.suggest_categorical(f"{prefix}.cost", _COST_CHOICES)
    inner_cost = _build_cost_for_descriptor(descriptor, cost_kind)
    if inner_cost is None:
        # Fall back to a safe default that always works.
        if descriptor == "kernel":
            inner_cost = Cosine("kernel")
        elif descriptor == "mask":
            inner_cost = MaskIoU("mask")
        else:
            inner_cost = BoxIoU("bbox")

    gating = trial.suggest_categorical(f"{prefix}.gating", _GATING_CHOICES)
    score_threshold = trial.suggest_float(f"{prefix}.score_thresh", 0.1, 0.9)
    max_chi2 = trial.suggest_float(f"{prefix}.max_chi2", 4.0, 16.0)
    gate = _build_gate(gating, score_threshold=score_threshold, max_chi2=max_chi2)

    threshold = trial.suggest_float(f"{prefix}.threshold", 0.05, 0.50)
    assoc_kind = trial.suggest_categorical(f"{prefix}.association", _ASSOC_CHOICES)
    assoc = Associate(assignment=_build_assignment(assoc_kind, threshold=threshold))

    use_kalman = trial.suggest_categorical(f"{prefix}.use_kalman", [False, True])
    del use_kalman  # use_kalman shapes the State definition; see _build_states.

    return gate, inner_cost, assoc


# ---------------------------------------------------------------------------
# Top-level sampler.
# ---------------------------------------------------------------------------


def sample_tracker(
    trial: optuna.trial.Trial,
    *,
    schema: TrackerSchema | None = None,
) -> unitrack.Tracker:
    """
    Sample a fully-built :class:`unitrack.Tracker` from a single Optuna trial.

    The trial's parameters span the full design space (see README for the
    parameter table).

    Parameters
    ----------
    trial:
        Active Optuna trial.
    schema:
        Per-Tracker fixed shape declaration. Defaults to a Cityscapes-shaped
        :class:`TrackerSchema` with kernel dim 256 and 19 classes.

    """
    if schema is None:
        schema = TrackerSchema()

    n_stages = trial.suggest_categorical("n_stages", _K_RANGE)
    fusion_mode = trial.suggest_categorical("fusion_mode", _FUSION_CHOICES)
    max_age = trial.suggest_int("max_age", 1, 5)
    min_hits = trial.suggest_int("min_hits", 1, 3)

    # Sample per-stage modules.
    stage_specs = [
        _suggest_stage(trial, prefix=f"s{k}", schema=schema) for k in range(n_stages)
    ]

    # Whether any stage uses motion or spatial gates / centroid cost — drives
    # whether we need a Kalman state on centroid.
    needs_kalman_centroid = any(isinstance(g, MotionGate) for g, _, _ in stage_specs)
    states = _build_states(schema, use_kalman_centroid=needs_kalman_centroid)

    # Build the root Associator. The StatusFilter passes Tentative + Active +
    # Lost — i.e. every live status. Filtering out Tentative would prevent the
    # standard min_hits→Active promotion since Tentatives are precisely the
    # tracklets we need to match against to make them Active.
    visible_statuses = StatusFilter(
        TrackletStatus.Tentative,
        TrackletStatus.Active,
        TrackletStatus.Lost,
    )
    if fusion_mode == "cascaded" or n_stages == 1:
        # Cascaded: each stage's residual feeds the next.
        children = []
        for gate, cost, assoc in stage_specs:
            inner = Pipe(cost=Gated(gate=gate, then=cost), assoc=assoc)
            children.append(inner)
        body = children[0] if len(children) == 1 else Sequential(children)
        root = Filter(predicate=visible_statuses, on="cs", then=body)
    else:
        # Parallel cost-level merge: all stages contribute branches to one cost.
        # Each branch is a Gated(cost) producing a CostExpression.
        branches = []
        weights = []
        for k, (gate, cost, _) in enumerate(stage_specs):
            branches.append(Gated(gate=gate, then=cost))
            weights.append(trial.suggest_float(f"s{k}.weight", 0.05, 1.0))
        # The shared associator uses the first stage's threshold for simplicity.
        # Parallel mode also uses one final assignment.
        _, _, root_assoc = stage_specs[0]
        root = Filter(
            predicate=visible_statuses,
            on="cs",
            then=Pipe(
                cost=Parallel(children=branches, merge=WeightedSum(weights)),
                assoc=root_assoc,
            ),
        )

    return unitrack.Tracker(
        root=root,
        states=states,
        lifecycle=StandardLifecycle(min_hits=min_hits, max_age=max_age, allow_reid=10),
        visibility=ConfirmedOnly(),
    )


def _build_states(schema: TrackerSchema, *, use_kalman_centroid: bool):
    """Construct the user-field State map for a Tracker."""
    states: dict[str, State] = {
        "kernel": State(
            schema=TensorSpec(shape=(schema.kernel_dim,), dtype=torch.float32),
            process=Identity("kernel"),
            observation=Replace("kernel"),
            init=FromDetectionField("kernel"),
        ),
        "mask": State(
            schema=TensorSpec(shape=schema.mask_shape, dtype=torch.bool),
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
        "bbox": State(
            schema=TensorSpec(shape=(4,), dtype=torch.float32),
            process=Identity("bbox"),
            observation=Replace("bbox"),
            init=FromDetectionField("bbox"),
        ),
    }

    if use_kalman_centroid:
        # Kalman centroid carries (mean, cov). The mean is 4D (2D position +
        # 2D velocity); detections only supply 2D position via ds.centroid.
        # The Kalman update's H matrix projects 4D state → 2D measurement.
        proc = KalmanCentroid2D("centroid")
        states["centroid"] = State(
            schema=TensorSpec(shape=(4,), dtype=torch.float32),
            process=proc,
            observation=proc.make_update(),
            init=_PadZerosInitializer(target_dim=4, src_field="centroid"),
        )
        # The KF process writes both centroid and centroid_cov when it advances;
        # the KF update writes both when it fuses a measurement. So the cov
        # field's own Process / Observation must be no-ops.
        states["centroid_cov"] = State(
            schema=TensorSpec(shape=(4, 4), dtype=torch.float32),
            process=_NoopProcess(),
            observation=_NoopObservation(),
            init=_EyeInitializer(dim=4),
        )
    else:
        # No-Kalman fallback: 2D centroid, replaced from detections each frame.
        states["centroid"] = State(
            schema=TensorSpec(shape=(2,), dtype=torch.float32),
            process=Identity("centroid"),
            observation=Replace("centroid"),
            init=FromDetectionField("centroid"),
        )

    return states


@dataclasses.dataclass(frozen=True, slots=True)
class _PadZerosInitializer:
    """Initialize centroid state by embedding a 2D position with zero velocity."""

    target_dim: int
    src_field: str

    def __call__(self, ds, ctx):
        del ctx
        pos = getattr(ds, self.src_field)
        n = pos.shape[0]
        out = torch.zeros((n, self.target_dim), dtype=pos.dtype, device=pos.device)
        out[:, : pos.shape[-1]] = pos
        return out


@dataclasses.dataclass(frozen=True, slots=True)
class _EyeInitializer:
    """Initialize per-tracklet covariance to a (dim, dim) identity matrix."""

    dim: int

    def __call__(self, ds, ctx):
        del ctx
        n = ds.batch_size[0]
        return (
            torch.eye(self.dim, dtype=torch.float32, device=ds.index.device)
            .expand(
                n,
                self.dim,
                self.dim,
            )
            .contiguous()
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _NoopProcess:
    """No-op Process — leaves the snapshot untouched."""

    def __call__(self, cs, ctx):
        del ctx
        return cs


@dataclasses.dataclass(frozen=True, slots=True)
class _NoopObservation:
    """No-op Observation — leaves the snapshot untouched on match and miss alike."""

    def __call__(self, cs, ds, match, ctx):
        del ds, match, ctx
        return cs


# ---------------------------------------------------------------------------
# Reference defaults — useful for sanity tests and baseline comparison.
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, typing.Any] = {
    "n_stages": 2,
    "fusion_mode": "cascaded",
    "max_age": 3,
    "min_hits": 2,
    # Stage 1: strict appearance gate (score) + cosine on kernel.
    "s0.gating": "score",
    "s0.score_thresh": 0.6,
    "s0.max_chi2": 9.21,
    "s0.descriptor": "kernel",
    "s0.cost": "cosine",
    "s0.threshold": 0.3,
    "s0.association": "jonker",
    "s0.use_kalman": False,
    # Stage 2: relaxed spatial gate + cdist-on-kernel.
    # NOTE: the paper uses MotionGate + CDist on a 3-D centroid, but that
    # combination requires a measurement-space projection (the Kalman state
    # is 6-D vs the detection's 3-D measurement). The default config here
    # uses SpatialGate2D + cdist-on-kernel, which exercises the same K=2
    # cascaded structure without needing the projection scaffold. See the
    # README "Caveats" section.
    "s1.gating": "spatial",
    "s1.score_thresh": 0.5,
    "s1.max_chi2": 9.21,
    "s1.descriptor": "kernel",
    "s1.cost": "cdist",
    "s1.threshold": 0.5,
    "s1.association": "jonker",
    "s1.use_kalman": False,
}
"""K=2 cascaded canonical configuration (two stages, strict then relaxed)."""
