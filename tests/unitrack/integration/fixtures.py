"""Deterministic synthetic clip fixture for paper-parity integration tests.

Generates a 20-frame detection sequence with 4 known-ground-truth objects
moving on cardinal trajectories. Two of the four objects briefly occlude
(missing detections) for one frame each, testing the tracker's re-id /
lifecycle handling. Each object carries:

- a distinct ``kernel`` (8-D embedding) so cosine similarity can
  distinguish identities;
- a deterministic ``centroid`` (3-D position) advancing per-frame so
  spatial / motion gates can be exercised;
- a class label (single class for now) and a constant high score.

A ``ground_truth`` array of shape ``(F, max_objects)`` carries the GT
object ID at each (frame, det_slot) pair (``-1`` for missing). This lets
the metrics module compute ID-switches, fragmentation, and a MOTA-like
quality score from the per-frame :class:`StepResult.match` outputs.

The fixture is deterministic — no randomness — so cached expected
metrics can pin tracker quality across refactors without flake.
"""

from __future__ import annotations

import dataclasses

import torch
from unitrack.data import Detections, FrameContext

__all__ = ["ClipFixture", "make_canonical_clip"]


@dataclasses.dataclass(frozen=True, slots=True)
class ClipFixture:
    """Per-frame deterministic detections + per-frame ground-truth ID labels."""

    detections: list[Detections]
    contexts: list[FrameContext]
    ground_truth: list[list[int]]  # ground_truth[f][det_slot] = GT object id, or -1


# Distinct unit embeddings for 4 objects (8-D), separated enough that
# cosine similarity is dominant over noise.
_OBJECT_EMBEDDINGS: torch.Tensor = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

# Per-frame, per-object trajectory (x, y, z). Each object moves along one
# cardinal direction over time. Occlusion frames (object absent) are
# encoded as None in the schedule.
_NUM_OBJECTS = 4


def _trajectories(num_frames: int) -> list[list[tuple[float, float, float] | None]]:
    """Return ``trajectories[frame][obj] = (x,y,z) or None`` (occlusion)."""
    schedule: list[list[tuple[float, float, float] | None]] = []
    for f in range(num_frames):
        per_frame: list[tuple[float, float, float] | None] = [None] * _NUM_OBJECTS
        per_frame[0] = (float(f), 0.0, 0.0)  # rightward
        per_frame[1] = (0.0, float(f), 0.0)  # upward
        per_frame[2] = (-float(f), 0.0, 0.0)  # leftward
        per_frame[3] = (0.0, -float(f), 0.0)  # downward
        # Occlusion holes: obj 0 missing on frame 7, obj 2 missing on frame 13.
        if f == 7:
            per_frame[0] = None
        if f == 13:
            per_frame[2] = None
        schedule.append(per_frame)
    return schedule


def _object_mask(obj_id: int) -> torch.Tensor:
    """Return a 4x4 boolean mask uniquely identifying ``obj_id`` (0..3).

    Each object owns a different 2x2 quadrant of the 4x4 mask so MaskIoU
    between same-object pairs is 1 (cost 0) and between different-object
    pairs is 0 (cost 1). Non-empty across the board so the "both-empty
    → cost 1" guard in MaskIoU doesn't trigger.
    """
    m = torch.zeros(4, 4, dtype=torch.bool)
    quadrants = [(0, 2, 0, 2), (0, 2, 2, 4), (2, 4, 0, 2), (2, 4, 2, 4)]
    r0, r1, c0, c1 = quadrants[obj_id % len(quadrants)]
    m[r0:r1, c0:c1] = True
    return m


def make_canonical_clip(num_frames: int = 20, fps: float = 15.0) -> ClipFixture:
    """Build the 20-frame canonical fixture."""
    traj = _trajectories(num_frames)
    detections: list[Detections] = []
    contexts: list[FrameContext] = []
    ground_truth: list[list[int]] = []

    for f, per_frame in enumerate(traj):
        present = [
            (obj_id, pos) for obj_id, pos in enumerate(per_frame) if pos is not None
        ]
        m = len(present)
        kernels: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        centroids: list[torch.Tensor] = []
        gt_row: list[int] = []
        for obj_id, pos in present:
            kernels.append(_OBJECT_EMBEDDINGS[obj_id])
            masks.append(_object_mask(obj_id))
            centroids.append(torch.tensor(pos, dtype=torch.float32))
            gt_row.append(obj_id)
        det = Detections(
            index=torch.arange(m, dtype=torch.int64),
            kernel=torch.stack(kernels) if kernels else torch.zeros(0, 8),
            mask=torch.stack(masks)
            if masks
            else torch.zeros(0, 4, 4, dtype=torch.bool),
            centroid=torch.stack(centroids) if centroids else torch.zeros(0, 3),
            klass=torch.zeros(m, dtype=torch.int64),
            score=torch.full((m,), 0.9, dtype=torch.float32),
            batch_size=[m],
        )
        detections.append(det)
        contexts.append(FrameContext.make(f, delta=1.0 / fps))
        ground_truth.append(gt_row)

    return ClipFixture(
        detections=detections, contexts=contexts, ground_truth=ground_truth
    )
