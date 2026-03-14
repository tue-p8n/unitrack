"""Render tracker output into an evaluators-compatible panoptic map."""

from __future__ import annotations

import numpy as np
import torch


class TrackRemap:
    """
    Stable, per-sequence ``(semantic_class, track_id) -> 1-based instance`` map.

    The panoptic encoding ``semantic * offset + instance`` requires the instance
    component to be a small positive integer (index 0 is reserved for
    crowd/void and is excluded by the metric). Track ids assigned by the tracker
    are arbitrary and may exceed ``offset``; this remaps them to ``1, 2, 3, …``
    in first-seen order, kept consistent across all frames of one sequence.

    Indexing is scoped per semantic class: two tracks of *different* classes may
    share an instance index without colliding in the encoding because their
    ``semantic`` component differs. A new index that would reach ``offset``
    (which would corrupt the encoding by carrying into the semantic component)
    raises a :class:`ValueError`.
    """

    def __init__(self, *, offset: int) -> None:
        self._offset = int(offset)
        self._map: dict[int, dict[int, int]] = {}

    def index_for(self, track_id: int, semantic_class: int) -> int:
        """Return the 1-based instance index for ``track_id`` in its class."""
        per_class = self._map.setdefault(semantic_class, {})
        idx = per_class.get(track_id)
        if idx is None:
            idx = len(per_class) + 1
            if idx >= self._offset:
                msg = (
                    f"instance index {idx} for semantic class {semantic_class} "
                    f"reaches offset {self._offset}; the panoptic encoding "
                    f"semantic*offset + instance would overflow"
                )
                raise ValueError(msg)
            per_class[track_id] = idx
        return idx


def render_pred_panoptic(  # noqa: PLR0913
    *,
    masks: torch.Tensor,
    categories: torch.Tensor,
    track_ids: torch.Tensor,
    height: int,
    width: int,
    offset: int,
    remap: TrackRemap,
) -> np.ndarray:
    """
    Paint thing instances as ``semantic * offset + dense_instance``.

    Detections with ``track_id < 0`` are skipped (a defensive contract; the
    runner only renders kept, tracked instances). Later masks win on overlap
    (model panoptic output is non-overlapping in practice).
    """
    pan = np.zeros((height, width), dtype=np.int64)
    masks_np = masks.cpu().numpy()
    cats = categories.cpu().tolist()
    tids = track_ids.cpu().tolist()
    for mask, cat, tid in zip(masks_np, cats, tids, strict=True):
        if tid < 0:
            continue
        inst = remap.index_for(int(tid), int(cat))
        pan[mask] = int(cat) * offset + inst
    return pan
