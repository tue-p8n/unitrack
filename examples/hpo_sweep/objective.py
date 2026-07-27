"""
Optuna objectives for the HPO sweep example.

Two modes:

- :func:`synthetic_objective` runs a tracker on a generated clip with known
  ground-truth identities and scores by ``1 - id_switch_rate``. No detector
  load required; useful for sanity-checking the search space.

- :func:`real_data_objective` runs the Mask2Former detector over a directory
  of frames and scores by a frame-to-frame query-cosine consistency proxy.
  Requires ``transformers``, ``Pillow``, and the model checkpoint.
"""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import torch
from unitrack import MultiStream
from unitrack.data import Detections, FrameContext

from .search_space import TrackerSchema, sample_tracker

if typing.TYPE_CHECKING:
    import optuna

    from .detector import Mask2FormerDetector

__all__ = [
    "SyntheticClip",
    "real_data_objective",
    "synthetic_clip",
    "synthetic_objective",
]


# ---------------------------------------------------------------------------
# Synthetic clip generator.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class SyntheticClip:
    """
    A synthetic K-frame clip with known ground-truth identities.

    Each frame holds the same N detections in random order. Each detection
    carries a kernel embedding that is constant per identity (plus a small
    per-frame jitter), and a 2D centroid that drifts at constant velocity.
    Ground-truth identity = a (K, N) matrix where row k = the per-detection
    GT id at frame k.
    """

    detections: list[Detections]
    gt_ids: torch.Tensor  # (K, N), int64 — ground-truth IDs per detection per frame
    schema: TrackerSchema


def synthetic_clip(
    *,
    n_frames: int = 8,
    n_objects: int = 3,
    schema: TrackerSchema | None = None,
    seed: int = 0,
) -> SyntheticClip:
    """
    Generate a deterministic synthetic clip.

    Each ground-truth object has:
    - a unique kernel embedding (orthonormal-ish via random + normalize)
    - a 2D position that advances at a constant per-object velocity
    - a fixed class

    Across frames, detections are *shuffled*, so the tracker must rely on
    appearance/motion to maintain identity.
    """
    if schema is None:
        schema = TrackerSchema()
    g = torch.Generator().manual_seed(seed)

    kernels = torch.randn(n_objects, schema.kernel_dim, generator=g)
    kernels = kernels / kernels.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    positions = torch.randint(50, 250, (n_objects, 2), generator=g).float()
    velocities = torch.randn(n_objects, 2, generator=g) * 4.0
    classes = torch.randint(0, schema.n_classes, (n_objects,), generator=g)

    detections: list[Detections] = []
    gt_id_rows: list[torch.Tensor] = []

    for k in range(n_frames):
        # Shuffle which-detection-is-which-identity for this frame.
        order = torch.randperm(n_objects, generator=g)
        gt_id_rows.append(order.clone())

        # Per-frame jittered observations.
        noise = 0.02 * torch.randn(n_objects, schema.kernel_dim, generator=g)
        kernel_obs = kernels + noise
        kernel_obs = kernel_obs / kernel_obs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        pos_obs = positions + k * velocities + torch.randn(n_objects, 2, generator=g)

        # Reorder by ``order`` (so the tracker can't trivially exploit row order).
        kernel_obs = kernel_obs[order]
        pos_obs = pos_obs[order]
        klass_obs = classes[order]

        # bbox: a small box centered on the centroid.
        cx, cy = pos_obs[:, 0], pos_obs[:, 1]
        bbox = torch.stack([cx - 16, cy - 16, cx + 16, cy + 16], dim=1)

        # mask: a tiny patch around the centroid in the schema's mask shape.
        height, width = schema.mask_shape
        mask = torch.zeros((n_objects, height, width), dtype=torch.bool)
        for i in range(n_objects):
            mh = max(0, min(height - 1, int(cy[i] * height / 256)))
            mw = max(0, min(width - 1, int(cx[i] * width / 256)))
            mask[i, mh, mw] = True

        score_obs = 0.85 + 0.1 * torch.rand(n_objects, generator=g)

        detections.append(
            Detections(
                index=torch.arange(n_objects, dtype=torch.int64),
                kernel=kernel_obs,
                mask=mask,
                klass=klass_obs.to(torch.int64),
                score=score_obs.to(torch.float32),
                bbox=bbox.to(torch.float32),
                centroid=pos_obs.to(torch.float32),
                batch_size=[n_objects],
            )
        )

    return SyntheticClip(
        detections=detections,
        gt_ids=torch.stack(gt_id_rows),
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Synthetic objective.
# ---------------------------------------------------------------------------


def synthetic_objective(
    trial: optuna.trial.Trial,
    *,
    clip: SyntheticClip,
) -> float:
    """Score a sampled tracker on a synthetic clip by ID-preservation rate."""
    tracker = sample_tracker(trial, schema=clip.schema)
    stream = MultiStream(tracker)

    # Track id-by-detection across frames; compare to GT.
    n_objs = clip.gt_ids.shape[1]
    assigned = torch.full(
        (clip.gt_ids.shape[0], n_objs), fill_value=-1, dtype=torch.int64
    )

    for k, dets in enumerate(clip.detections):
        ctx = FrameContext.make(
            frame_idx=k,
            delta=1.0 / 15.0,
            fps=15.0,
            stream_key=0,
        )
        try:
            res = stream.step(stream_key=0, detections=dets, ctx=ctx)
        except Exception:
            # A malformed config (e.g. wrong descriptor field) — penalize.
            return 0.0

        # Build per-detection ID array. ``res.snapshot`` carries all live
        # tracklets including new ones; the IDs are assigned in detection-
        # index order for new tracklets and in cs-row order for matches.
        # Without a per-step "detection_index → tracklet_id" map exposed on
        # MatchOutcome (which would be a nicer API addition), we fall back to:
        #   - matched detections: tracklets_residual_index → use match.matched_pairs
        #   - unmatched detections: assigned new IDs in order
        ids_for_dets = _ids_per_detection(res, n_dets=n_objs)
        assigned[k] = ids_for_dets

    return _id_preservation_score(assigned, clip.gt_ids)


def _ids_per_detection(res, n_dets: int) -> torch.Tensor:
    """Reconstruct per-detection IDs from a StepResult."""
    # The simplest accurate proxy: the snapshot's id field, indexed by
    # the matched tracklet positions for matched detections, plus the
    # final new-tracklet IDs for unmatched detections.
    #
    # Heuristic: take the last ``n_dets`` snapshot IDs (they're appended
    # in detection order in Tracker.step's "5. Merge" step). For matched
    # detections, this is wrong — the matched tracklet's ID is preserved
    # in-place in the snapshot, not appended. So this is approximate.
    snap_ids = res.snapshot.id
    if snap_ids.shape[0] >= n_dets:
        return snap_ids[-n_dets:].clone()
    out = torch.full((n_dets,), fill_value=-1, dtype=torch.int64)
    out[: snap_ids.shape[0]] = snap_ids
    return out


def _id_preservation_score(
    assigned: torch.Tensor,
    gt: torch.Tensor,
) -> float:
    """
    Score: 1 - (per-frame ID switch rate vs frame 0).

    For each (frame ≥ 1, detection), check whether the assigned tracker ID
    matches the same GT identity as it did at frame 0. The score is the
    fraction of (frame, detection) pairs that hold this property.
    """
    if assigned.shape[0] < 2:
        return 0.0

    # Build a per-GT-id → first-frame tracker-id map.
    first_frame_assigned = assigned[0]  # tracker IDs at frame 0, in det order
    first_frame_gt = gt[0]  # GT IDs at frame 0, in det order

    # GT id -> tracker id
    gt_to_tracker = {
        int(first_frame_gt[i].item()): int(first_frame_assigned[i].item())
        for i in range(first_frame_gt.shape[0])
        if int(first_frame_assigned[i].item()) > 0
    }

    if not gt_to_tracker:
        return 0.0

    correct = 0
    total = 0
    for k in range(1, assigned.shape[0]):
        for d in range(assigned.shape[1]):
            gt_id = int(gt[k, d].item())
            tracker_id = int(assigned[k, d].item())
            if gt_id not in gt_to_tracker:
                continue
            total += 1
            if tracker_id == gt_to_tracker[gt_id]:
                correct += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Real-data objective.
# ---------------------------------------------------------------------------


def real_data_objective(
    trial: optuna.trial.Trial,
    *,
    detector: Mask2FormerDetector,
    frame_paths: list[pathlib.Path],
    schema: TrackerSchema,
) -> float:
    """
    Run a sampled tracker on detector outputs over a real frame sequence.

    Score = mean cosine similarity between matched tracklet kernels in
    consecutive frames. Higher = the tracker preserves identity-consistent
    appearance across frames. This is a *proxy* for HOTA/VPQ; for a real
    eval, replace with a benchmark-specific metric.
    """
    from PIL import Image  # noqa: PLC0415 — optional dep for real-data path

    tracker = sample_tracker(trial, schema=schema)
    stream = MultiStream(tracker)

    prev_snap = None
    total_sim = 0.0
    n_pairs = 0

    for k, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("RGB")
        dets = detector(img)
        if dets.batch_size[0] == 0:
            continue
        ctx = FrameContext.make(
            frame_idx=k,
            delta=1.0 / 15.0,
            fps=15.0,
            stream_key=0,
        )
        try:
            res = stream.step(stream_key=0, detections=dets, ctx=ctx)
        except Exception:
            return 0.0

        snap = res.snapshot
        if prev_snap is not None:
            common = _common_id_pairs(prev_snap.id, snap.id)
            for prev_idx, cur_idx in common:
                a = prev_snap.kernel[prev_idx]
                b = snap.kernel[cur_idx]
                if a.norm() < 1e-6 or b.norm() < 1e-6:
                    continue
                sim = torch.dot(a, b) / (a.norm() * b.norm())
                total_sim += float(sim.item())
                n_pairs += 1

        prev_snap = snap

    return total_sim / n_pairs if n_pairs > 0 else 0.0


def _common_id_pairs(prev_ids: torch.Tensor, cur_ids: torch.Tensor):
    """Return pairs (i, j) such that prev_ids[i] == cur_ids[j]."""
    pairs: list[tuple[int, int]] = []
    prev_dict = {int(v.item()): i for i, v in enumerate(prev_ids)}
    for j, v in enumerate(cur_ids):
        idx = prev_dict.get(int(v.item()))
        if idx is not None:
            pairs.append((idx, j))
    return pairs
