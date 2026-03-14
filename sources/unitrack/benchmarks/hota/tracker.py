"""
Tracker factories (mask-IoU, cosine, cascade, Kalman) + a registry.

Each factory builds a :class:`~unitrack.Tracker` over a fixed ``(height, width)``
from existing unitrack primitives — no core-library changes. ``TRACKER_REGISTRY``
maps a key to a ``(height, width) -> MultiStream`` factory so the benchmark can
sweep the tracker as a first-class variable, parallel to the model registry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

import unitrack
from unitrack import MultiStream
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine, Mahalanobis, MaskIoU
from unitrack.data import TensorSpec
from unitrack.gates import ClassGate, MotionGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Filter, Gated, Pipe, Sequential
from unitrack.states import (
    FromDetectionField,
    Identity,
    LearnedObservation,
    LearnedProcess,
    Replace,
    State,
)
from unitrack.states.kalman import KalmanCentroid2D
from unitrack.tracker import StepResult

from .learned_modules import Fuser, Propagator

# Mask2Former decoder-query / appearance-embedding dimensionality.
EMBED_DIM = 256

# Default committed checkpoint for the learned filter.
DEFAULT_LEARNED_CKPT = Path(__file__).parent / "weights" / "learned_filter.safetensors"

# 0.95 quantile of the chi-squared distribution at 2 d.o.f. (centroid x, y),
# the standard SORT/DeepSORT motion-gate threshold.
CHI2_GATE = 5.9915


def build_mask_tracker(
    *,
    height: int,
    width: int,
    cost_threshold: float = 0.5,
    class_gate: bool = True,
) -> unitrack.Tracker:
    """
    Build a single-stage mask-IoU tracker over a fixed ``(height, width)``.

    ``cost_threshold`` is the maximum acceptable ``1 - IoU`` association cost,
    i.e. matches require ``IoU >= 1 - cost_threshold``. ``NoLifecycle`` keeps
    every track (no row drops), which ``ids_per_detection`` relies on.

    Score filtering is not done here: low-confidence detections are dropped in
    the runner before a ``Detections`` is built, so the tracker only ever sees
    kept instances and carries no ``score`` state.
    """
    inner = Pipe(
        cost=MaskIoU("mask"), assoc=Associate(Jonker(threshold=cost_threshold))
    )
    gates: list = []
    if class_gate:
        gates.append(ClassGate("category"))
    pipeline = Gated(gate=Sequential(gates), then=inner)
    return unitrack.Tracker(
        root=pipeline,
        states={
            "mask": State(
                schema=TensorSpec(shape=(height, width), dtype=torch.bool),
                process=Identity("mask"),
                observation=Replace("mask"),
                init=FromDetectionField("mask"),
            ),
            "category": State(
                schema=TensorSpec(shape=(), dtype=torch.int64),
                process=Identity("category"),
                observation=Replace("category"),
                init=FromDetectionField("category"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


def default_tracker_factory(cost_threshold: float = 0.5):
    """
    Return a :class:`TrackerFactory` building a fresh mask-IoU ``MultiStream``.

    The returned callable takes the frame ``(height, width)`` (known only once
    a sequence's first frame is seen) and wraps :func:`build_mask_tracker`.
    """

    def factory(height: int, width: int) -> MultiStream:
        return MultiStream(
            build_mask_tracker(
                height=height, width=width, cost_threshold=cost_threshold
            )
        )

    return factory


def _embedding_state(dim: int, *, process=None, observation=None) -> State:
    """
    Build an appearance-embedding state (``(dim,)`` float32).

    Defaults to no prediction (``Identity``) and hard-replace on match
    (``Replace``); the learned filter supplies a ``LearnedProcess`` /
    ``LearnedObservation`` pair instead.
    """
    return State(
        schema=TensorSpec(shape=(dim,), dtype=torch.float32),
        process=process or Identity("embedding"),
        observation=observation or Replace("embedding"),
        init=FromDetectionField("embedding"),
    )


def _category_state() -> State:
    """Build the int64 scalar semantic-class state (used by ``ClassGate``)."""
    return State(
        schema=TensorSpec(shape=(), dtype=torch.int64),
        process=Identity("category"),
        observation=Replace("category"),
        init=FromDetectionField("category"),
    )


def _score_state() -> State:
    """Build the float32 scalar confidence state (carried for the cascade)."""
    return State(
        schema=TensorSpec(shape=(), dtype=torch.float32),
        process=Identity("score"),
        observation=Replace("score"),
        init=FromDetectionField("score"),
    )


def _mask_state(height: int, width: int) -> State:
    """Build the boolean ``(height, width)`` instance-mask state."""
    return State(
        schema=TensorSpec(shape=(height, width), dtype=torch.bool),
        process=Identity("mask"),
        observation=Replace("mask"),
        init=FromDetectionField("mask"),
    )


def build_cosine_tracker(
    *,
    height: int,
    width: int,
    cost_threshold: float = 0.5,
    embed_dim: int = EMBED_DIM,
) -> unitrack.Tracker:
    """
    Build an appearance tracker matching instances by embedding cosine distance.

    Matches require ``1 - cosine_similarity <= cost_threshold`` within the same
    class. ``height`` / ``width`` are accepted for a uniform factory signature
    but unused (appearance matching needs no mask state).
    """
    del height, width
    inner = Pipe(
        cost=Cosine("embedding"),
        assoc=Associate(Jonker(threshold=cost_threshold)),
    )
    pipeline = Gated(gate=ClassGate("category"), then=inner)
    return unitrack.Tracker(
        root=pipeline,
        states={
            "embedding": _embedding_state(embed_dim),
            "category": _category_state(),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


def build_cascade_tracker(  # noqa: PLR0913
    *,
    height: int,
    width: int,
    hi: float = 0.9,
    cos_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    embed_dim: int = EMBED_DIM,
) -> unitrack.Tracker:
    """
    Build a two-stage cascade: appearance for high-score, mask-IoU for the rest.

    Within a class, detections split by score. High-confidence detections
    (``score >= hi``) go only to the embedding-cosine stage; a high-score
    detection left unmatched there spawns a new track (it is NOT re-offered to
    mask-IoU). Low-confidence detections (``score < hi``) go only to the mask-IoU
    stage. ``hi`` is the cascade's internal split, independent of the runner's
    ``min_score`` floor.

    The default ``hi=0.9`` is tuned for panoptic-segmentation scores, which
    cluster high (Mask2Former Cityscapes thing scores have a ~0.8 floor and a
    ~0.997 median): a lower split would route every detection to the appearance
    stage and the cascade would collapse to the pure-cosine tracker.
    """
    # The class gate is applied inside each stage's Pipe (a per-pair gate needs
    # a cost to bias, so it must wrap the Pipe directly, not the cascade).
    hi_stage = Filter(
        predicate=lambda ds: ds.score >= hi,
        then=Gated(
            gate=ClassGate("category"),
            then=Pipe(
                cost=Cosine("embedding"),
                assoc=Associate(Jonker(threshold=cos_threshold)),
            ),
        ),
        on="ds",
    )
    lo_stage = Filter(
        predicate=lambda ds: ds.score < hi,
        then=Gated(
            gate=ClassGate("category"),
            then=Pipe(
                cost=MaskIoU("mask"),
                assoc=Associate(Jonker(threshold=iou_threshold)),
            ),
        ),
        on="ds",
    )
    pipeline = Sequential([hi_stage, lo_stage])
    return unitrack.Tracker(
        root=pipeline,
        states={
            "embedding": _embedding_state(embed_dim),
            "mask": _mask_state(height, width),
            "category": _category_state(),
            "score": _score_state(),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


def build_kalman_tracker(  # noqa: PLR0913
    *,
    height: int,
    width: int,
    max_chi2: float = CHI2_GATE,
    jonker_threshold: float | None = None,
    q: float = 1.0,
    r: float = 1.0,
) -> unitrack.Tracker:
    """
    Build a constant-velocity centroid Kalman tracker with a motion gate.

    The tracklet centroid state is 4-D ``[x, y, vx, vy]`` (predicted forward
    each frame); detections carry a 2-D ``[x, y]`` centroid. The association
    cost is the Mahalanobis chi-squared distance to the *predicted* centroid
    (projected into the 2-D measurement subspace), gated by ``MotionGate`` at
    ``max_chi2`` and ``ClassGate``. A plain ``CDist("centroid")`` cannot be used
    here: the 4-D tracklet mean and 2-D detection mean have mismatched feature
    dimensions; ``Mahalanobis`` is the primitive that projects between them.

    ``max_chi2`` bounds the ``MotionGate`` (pairs with a larger chi-squared
    distance are masked out before assignment). ``jonker_threshold`` bounds the
    Jonker assignment cost and defaults to ``max_chi2`` (so the gate and the
    assignment share one rejection radius, the standard SORT setting); passing a
    looser ``jonker_threshold`` isolates the ``MotionGate`` as the sole cause of a
    rejection, which the gate-isolation test relies on.

    ``height`` / ``width`` are accepted for a uniform factory signature but the
    Kalman state is resolution-independent (pixel centroids). ``q`` / ``r`` are
    sized for pixel-space centroids (process/measurement noise of order one
    pixel²), not the normalized-coordinate defaults of ``KalmanCentroid2D``.
    """
    del height, width
    if jonker_threshold is None:
        jonker_threshold = max_chi2
    kal = KalmanCentroid2D(field="centroid", q=q, r=r)
    centroid_states = kal.state_entries(meas_field="centroid")
    gate = Sequential(
        [
            ClassGate("category"),
            MotionGate(
                mean_field="centroid", cov_field="centroid_cov", max_chi2=max_chi2
            ),
        ]
    )
    inner = Pipe(
        cost=Mahalanobis("centroid", "centroid_cov"),
        assoc=Associate(Jonker(threshold=jonker_threshold)),
    )
    pipeline = Gated(gate=gate, then=inner)
    return unitrack.Tracker(
        root=pipeline,
        states={**centroid_states, "category": _category_state()},
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


def _load_learned_modules(
    checkpoint: str | Path, embed_dim: int
) -> tuple[Propagator, Fuser]:
    """
    Load the trained Propagator/Fuser pair from a flattened safetensors file.

    Raises a clear :class:`FileNotFoundError` pointing at the training script when
    the checkpoint is absent — the learned tracker is never silently untrained.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        msg = (
            f"learned-filter checkpoint not found at {checkpoint}; run "
            "benchmarks/hota/train_learned.py (python -m "
            "unitrack.benchmarks.hota.train_learned) to produce it"
        )
        raise FileNotFoundError(msg)
    from safetensors.torch import load_file

    flat = load_file(str(checkpoint))
    prop_sd = {
        k[len("propagator.") :]: v
        for k, v in flat.items()
        if k.startswith("propagator.")
    }
    fuse_sd = {k[len("fuser.") :]: v for k, v in flat.items() if k.startswith("fuser.")}
    prop = Propagator(embed_dim)
    fuse = Fuser(embed_dim)
    prop.load_state_dict(prop_sd)
    fuse.load_state_dict(fuse_sd)
    prop.eval()
    fuse.eval()
    return prop, fuse


def build_learned_tracker(
    *,
    height: int,
    width: int,
    cost_threshold: float = 0.5,
    embed_dim: int = EMBED_DIM,
    checkpoint: str | Path = DEFAULT_LEARNED_CKPT,
) -> unitrack.Tracker:
    """
    Build a MOTR-style learned appearance tracker from a trained checkpoint.

    Wires the same cosine-distance association as :func:`build_cosine_tracker`,
    but the embedding state is filtered by learned modules: a
    :class:`~.learned_modules.Propagator` (predict, via ``LearnedProcess``) and a
    :class:`~.learned_modules.Fuser` (update on match, via
    ``LearnedObservation``), loaded from ``checkpoint``. Raises
    :class:`FileNotFoundError` if the checkpoint is missing.

    ``height`` / ``width`` are accepted for a uniform factory signature but unused
    (appearance matching needs no mask state).
    """
    del height, width
    prop, fuse = _load_learned_modules(checkpoint, embed_dim)
    inner = Pipe(
        cost=Cosine("embedding"),
        assoc=Associate(Jonker(threshold=cost_threshold)),
    )
    pipeline = Gated(gate=ClassGate("category"), then=inner)
    return unitrack.Tracker(
        root=pipeline,
        states={
            "embedding": _embedding_state(
                embed_dim,
                process=LearnedProcess("embedding", prop),
                observation=LearnedObservation("embedding", "embedding", fuse),
            ),
            "category": _category_state(),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


TrackerFactory = Callable[[int, int], MultiStream]


def _wrap(build, **kwargs) -> TrackerFactory:
    """Wrap a ``build_*`` callable into a ``(height, width) -> MultiStream``."""

    def factory(height: int, width: int) -> MultiStream:
        return MultiStream(build(height=height, width=width, **kwargs))

    return factory


TRACKER_REGISTRY: dict[str, TrackerFactory] = {
    "maskiou": _wrap(build_mask_tracker),
    "cosine": _wrap(build_cosine_tracker),
    "cascade": _wrap(build_cascade_tracker),
    "kalman": _wrap(build_kalman_tracker),
    "learned": _wrap(build_learned_tracker),
}


def ids_per_detection(res: StepResult, n_dets: int) -> torch.Tensor:
    """
    Recover per-detection track ids in detection order (``-1`` if gated out).

    Accurate under ``NoLifecycle``: ``snapshot`` is ``cat([updated, spawned])``
    with no rows dropped, so matched detections read their id from
    ``snapshot.id[matched_pairs[:, 0]]`` and unmatched (residual) detections read
    the appended rows in residual order.
    """
    match = res.match
    snap_ids = res.snapshot.id
    out = torch.full((n_dets,), -1, dtype=torch.int64, device=snap_ids.device)
    pairs = match.matched_pairs
    if pairs.numel():
        out[pairs[:, 1]] = snap_ids[pairs[:, 0]]
    residual = match.detections_residual_index
    n_new = int(residual.shape[0])
    if n_new:
        n_pred = int(snap_ids.shape[0]) - n_new
        out[residual] = snap_ids[n_pred:]
    return out
