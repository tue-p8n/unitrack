"""Wrap the evaluators panoptic_tracking() preset as a streaming runner."""

from __future__ import annotations

import numpy as np


class PanopticMetricRunner:
    """
    Stream (gt, pred) panoptic maps into HOTA/CLEAR/Identity and aggregate.

    Surfaces a flat ``{metric_name: float}`` of the headline scores: HOTA/DetA/
    AssA/LocA (``___AUC`` aggregates) plus MOTA and IDF1.
    """

    def __init__(self, *, offset: int, thing_ids: list[int]) -> None:
        from evaluators.presets.mot import panoptic_tracking

        self._metrics = panoptic_tracking(offset=offset, thing_ids=thing_ids)
        for m in self._metrics:
            m.reset()

    def start_sequence(self, *, length: int) -> None:
        """Open a new sequence of ``length`` frames on every metric."""
        for m in self._metrics:
            m.on_sequence_start(length)

    def update(self, *, gt_panoptic: np.ndarray, pred_panoptic: np.ndarray) -> None:
        """Feed one frame's ground-truth and predicted panoptic maps."""
        for m in self._metrics:
            m.update(gt_pan=gt_panoptic, pred_pan=pred_panoptic)

    def end_sequence(self) -> None:
        """Close the current sequence on every metric."""
        for m in self._metrics:
            m.on_sequence_end()

    def compute(self) -> dict[str, float]:
        """Aggregate the streamed sequences into headline scalar scores."""
        hota, clear, identity = (m.compute() for m in self._metrics)
        out: dict[str, float] = {}
        for key in ("HOTA", "DetA", "AssA", "LocA"):
            out[key] = float(hota[f"{key}___AUC"])
        out["MOTA"] = float(clear["MOTA"])
        out["IDF1"] = float(identity["IDF1"])
        return out
