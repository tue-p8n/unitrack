"""Tracking metrics: ID-switches + ID-consistency rate.

For each ground-truth object, we trace the sequence of tracker IDs the
tracker assigned to that object across frames. A clean tracker keeps
each GT on a single tracker ID across the whole clip; an ID switch is
any change in the assigned tracker ID when the GT was visible in
consecutive (or near-consecutive) frames.

These metrics are what "paper parity" means for the synthetic canonical
fixture — a real-data MOTA / IDF1 benchmark against published numbers
needs an actual detector and lives in the paper repo.

Per-detection → tracker-ID mapping
----------------------------------
After ``tracker.step``, every detection ends up associated with one row
in ``result.snapshot``:

- If detection ``k`` appears in ``match.matched_pairs[:, 1]`` at position
  ``j``, the merged-row index is ``match.matched_pairs[j, 0]``.
- Otherwise detection ``k`` spawned a new tracklet; the new rows are
  appended to the merged snapshot in the order of
  ``match.detections_residual_index``. So if ``k`` appears at position
  ``i`` in that index, the merged-row index is ``n_predicted + i``.

For ``NoLifecycle`` the merged snapshot is the returned snapshot, so the
merged-row index is also the snapshot-row index, and the tracker ID is
``snapshot.id[snapshot_row]``.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class TrackingMetrics:
    """Aggregate tracking metrics across a clip.

    ``misses`` counts detections present in the ground truth that the
    tracker did not associate with any row in the post-step snapshot
    (gated out, dropped by lifecycle, or out-of-range residual index).
    A clean run on ``NoLifecycle`` with the canonical fixture should
    report ``misses == 0``.
    """

    total_observations: int
    id_switches: int
    consistency_rate: float
    misses: int


def _per_detection_tracker_ids(res, n_predicted_pre_step: int) -> dict[int, int]:
    """
    Return ``det_idx → tracker_id`` for every detection in this frame.

    ``n_predicted_pre_step`` is the snapshot row count *before* the step
    ran — needed to compute where newly-spawned tracklets land in the
    post-step row ordering.
    """
    snap = res.snapshot
    match = res.match
    out: dict[int, int] = {}

    # Matched detections: pair[j] = (merged_row, det_idx)
    if match.matched_pairs.shape[0] > 0:
        for j in range(match.matched_pairs.shape[0]):
            merged_row = int(match.matched_pairs[j, 0].item())
            det_idx = int(match.matched_pairs[j, 1].item())
            if merged_row < snap.batch_size[0]:
                out[det_idx] = int(snap.id[merged_row].item())

    # Unmatched detections: spawned new tracklets appended in order of
    # `match.detections_residual_index`.
    residual = match.detections_residual_index
    for i in range(residual.shape[0]):
        det_idx = int(residual[i].item())
        snap_row = n_predicted_pre_step + i
        if snap_row < snap.batch_size[0]:
            out[det_idx] = int(snap.id[snap_row].item())

    return out


def compute_metrics(
    step_results: list,
    ground_truth: list[list[int]],
    initial_snapshot_size: int = 0,
) -> TrackingMetrics:
    """Compute :class:`TrackingMetrics` from per-frame step results."""
    total_observations = 0
    id_switches = 0
    misses = 0
    last_tracker_id: dict[int, int] = {}

    n_predicted = initial_snapshot_size
    for gt_row, res in zip(ground_truth, step_results, strict=True):
        det_to_tracker = _per_detection_tracker_ids(res, n_predicted)
        # For next frame: pre-step snapshot size equals this frame's
        # post-step snapshot size (NoLifecycle preserves rows).
        n_predicted = res.snapshot.batch_size[0]

        for det_idx, gt_id in enumerate(gt_row):
            tid = det_to_tracker.get(det_idx)
            if tid is None:
                misses += 1
                continue
            total_observations += 1
            prev = last_tracker_id.get(gt_id)
            if prev is not None and prev != tid:
                id_switches += 1
            last_tracker_id[gt_id] = tid

    consistency_rate = 1.0 - id_switches / max(total_observations, 1)
    return TrackingMetrics(
        total_observations=total_observations,
        id_switches=id_switches,
        consistency_rate=consistency_rate,
        misses=misses,
    )
