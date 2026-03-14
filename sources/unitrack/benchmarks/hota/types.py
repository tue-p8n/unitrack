"""Plain data carriers exchanged between the harness plug points."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import torch


@dataclasses.dataclass(frozen=True, slots=True)
class FramePrediction:
    """
    One frame of model output: per-instance thing masks + semantics.

    Attributes
    ----------
    masks
        ``(M, H, W)`` boolean instance masks.
    categories
        ``(M,)`` int64 semantic class id per instance (dataset label space).
    scores
        ``(M,)`` float32 confidence per instance.
    embeddings
        Optional ``(M, D)`` appearance/query embeddings (unused by the default
        mask-IoU tracker; available for appearance-augmented variants).
    centroids
        Optional ``(M, 2)`` per-instance mask centroids ``(x, y)`` (unused by the
        default mask-IoU tracker; available for motion-augmented variants).

    """

    masks: torch.Tensor
    categories: torch.Tensor
    scores: torch.Tensor
    embeddings: torch.Tensor | None = None
    centroids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate that masks/categories/scores agree on row count and rank."""
        m = self.masks.shape[0]
        if self.categories.shape[0] != m or self.scores.shape[0] != m:
            msg = "FramePrediction row counts disagree (masks/categories/scores)"
            raise ValueError(msg)
        if self.masks.ndim != 3:
            msg = "masks must be (M, H, W)"
            raise ValueError(msg)
        if self.embeddings is not None and self.embeddings.shape[0] != m:
            msg = "FramePrediction embeddings row count disagrees with masks"
            raise ValueError(msg)
        if self.centroids is not None and (
            self.centroids.shape[0] != m
            or self.centroids.ndim != 2
            or self.centroids.shape[1] != 2
        ):
            msg = "FramePrediction centroids must be (M, 2) matching the mask count"
            raise ValueError(msg)

    @property
    def num_instances(self) -> int:
        """Number of predicted instances ``M``."""
        return int(self.masks.shape[0])

    @property
    def height(self) -> int:
        """Mask height ``H``."""
        return int(self.masks.shape[1])

    @property
    def width(self) -> int:
        """Mask width ``W``."""
        return int(self.masks.shape[2])


@dataclasses.dataclass(frozen=True, slots=True)
class SequenceSample:
    """
    One ground-truth sequence: an id, length, and a frame iterator.

    Each item from ``frames`` is ``(image, gt_panoptic)`` where ``image`` is
    ``(H, W, 3)`` uint8 RGB and ``gt_panoptic`` is ``(H, W)`` int64 encoded as
    ``semantic * offset + instance``.
    """

    sequence_id: str
    length: int
    frames: Iterator[tuple[np.ndarray, np.ndarray]]


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One ``(model, tracker)`` combo's aggregated metric output plus timing."""

    model_key: str
    tracker_key: str
    metrics: dict[str, float]
    num_sequences: int
    num_frames: int
    seconds: float
