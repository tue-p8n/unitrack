"""Sweep models x sequences, streaming predictions into the HOTA metric."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from unitrack.data import Detections, FrameContext

from .metric import PanopticMetricRunner
from .protocols import DatasetAdapter, ModelAdapter, TrackerFactory
from .render import TrackRemap, render_pred_panoptic
from .tracker import default_tracker_factory, ids_per_detection
from .types import BenchmarkResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

    from unitrack import MultiStream


class BenchmarkRunner:
    """Run each model over every sequence and aggregate HOTA per model."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        device: torch.device | None = None,
        cost_threshold: float = 0.5,
        min_score: float = 0.1,
        fps: float = 15.0,
        tracker_factory: TrackerFactory | None = None,
        metric_factory: Callable[[int, list[int]], PanopticMetricRunner] | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.cost_threshold = cost_threshold
        self.min_score = min_score
        self.fps = fps
        self.tracker_factory = tracker_factory or default_tracker_factory(
            cost_threshold=cost_threshold
        )
        self.metric_factory = metric_factory or (
            lambda offset, thing_ids: PanopticMetricRunner(
                offset=offset, thing_ids=thing_ids
            )
        )

    def run(
        self,
        *,
        models: list[ModelAdapter],
        dataset: DatasetAdapter,
        trackers: dict[str, TrackerFactory] | None = None,
    ) -> list[BenchmarkResult]:
        """
        Score every ``(model, tracker)`` combo on ``dataset``, one result each.

        ``trackers`` maps a tracker key to its factory; when omitted it defaults
        to ``{"maskiou": <constructor tracker_factory>}`` so the single-tracker
        behavior is preserved. A combo that fails (e.g. an unsupported checkpoint,
        an out-of-memory load, or a tracker that cannot be built) is logged and
        skipped so it cannot lose the rest of the sweep; only successfully-scored
        combos appear in the returned list.
        """
        if trackers is None:
            trackers = {"maskiou": self.tracker_factory}
        results: list[BenchmarkResult] = []
        for model in models:
            for tracker_key, tracker_factory in trackers.items():
                try:
                    results.append(
                        self._run_one(model, dataset, tracker_factory, tracker_key)
                    )
                except Exception:
                    logger.exception(
                        "benchmark model %r x tracker %r failed; skipping",
                        getattr(model, "key", "<unknown>"),
                        tracker_key,
                    )
        return results

    def _run_one(
        self,
        model: ModelAdapter,
        dataset: DatasetAdapter,
        tracker_factory: TrackerFactory,
        tracker_key: str,
    ) -> BenchmarkResult:
        model.load(self.device)
        metric = self.metric_factory(dataset.offset, list(dataset.thing_ids))
        n_seq = n_frames = 0
        t0 = time.perf_counter()
        for sample in dataset.sequences():
            n_seq += 1
            metric.start_sequence(length=sample.length)
            # One fresh tracker per sequence, built lazily on the first frame
            # once the sequence's (height, width) is known.
            tracker: MultiStream | None = None
            remap = TrackRemap(offset=dataset.offset)
            for frame_idx, (image, gt_panoptic) in enumerate(sample.frames):
                h, w = gt_panoptic.shape
                if tracker is None:
                    tracker = tracker_factory(h, w)
                pred_panoptic = self._predict_panoptic(
                    model=model,
                    image=image,
                    tracker=tracker,
                    remap=remap,
                    offset=dataset.offset,
                    height=h,
                    width=w,
                    frame_idx=frame_idx,
                )
                metric.update(gt_panoptic=gt_panoptic, pred_panoptic=pred_panoptic)
                n_frames += 1
            metric.end_sequence()
        seconds = time.perf_counter() - t0
        return BenchmarkResult(
            model_key=model.key,
            tracker_key=tracker_key,
            metrics=metric.compute(),
            num_sequences=n_seq,
            num_frames=n_frames,
            seconds=seconds,
        )

    def _predict_panoptic(  # noqa: PLR0913
        self,
        *,
        model: ModelAdapter,
        image: np.ndarray,
        tracker: MultiStream,
        remap: TrackRemap,
        offset: int,
        height: int,
        width: int,
        frame_idx: int,
    ) -> np.ndarray:
        """Run one frame: predict, score-filter, track, and render a panoptic."""
        pred = model(image)
        keep = pred.scores >= self.min_score
        masks = pred.masks[keep]
        categories = pred.categories[keep]
        n_dets = int(masks.shape[0])
        dev = self.device
        # ``score`` is always carried (cheap; the cascade tracker filters on it);
        # ``embedding`` / ``centroid`` only when the model produced them, so the
        # mask-IoU path stays bit-identical to its pre-embedding behavior.
        fields = {
            "mask": masks.to(dev),
            "category": categories.to(dev),
            "score": pred.scores[keep].to(dev),
        }
        if pred.embeddings is not None:
            fields["embedding"] = pred.embeddings[keep].to(dev)
        if pred.centroids is not None:
            fields["centroid"] = pred.centroids[keep].to(dev)
        dets = Detections(
            index=torch.arange(n_dets, device=dev),
            batch_size=[n_dets],
            **fields,
        )
        ctx = FrameContext.make(frame_idx, fps=self.fps, stream_key=0)
        res = tracker.step(0, dets, ctx)
        track_ids = ids_per_detection(res, n_dets=n_dets)
        return render_pred_panoptic(
            masks=masks,
            categories=categories,
            track_ids=track_ids,
            height=height,
            width=width,
            offset=offset,
            remap=remap,
        )
