"""In-memory fakes for CI-safe harness tests (no network / no GPU)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from unitrack.benchmarks.hota.types import FramePrediction, SequenceSample

_OFFSET = 1000
_THING_IDS = [11]
_THING_IDS_SPLIT = [11, 13]


def _rect_mask(h, w, box) -> torch.Tensor:
    t = torch.zeros((h, w), dtype=torch.bool)
    y0, x0, y1, x1 = box
    t[y0:y1, x0:x1] = True
    return t


class SyntheticPanopticDataset:
    """Two 8x8 frames with one perfectly-consistent class-11 thing instance."""

    key = "synthetic"
    thing_ids = _THING_IDS
    offset = _OFFSET

    def __init__(self, *, n_frames: int = 2) -> None:
        self.n_frames = n_frames

    def sequences(self) -> Iterator[SequenceSample]:
        h = w = 8
        gt = np.zeros((h, w), dtype=np.int64)
        gt[0:4, 0:4] = 11 * _OFFSET + 1  # instance id 1, stable across frames
        frames = ((np.zeros((h, w, 3), np.uint8), gt) for _ in range(self.n_frames))
        yield SequenceSample(sequence_id="000000", length=self.n_frames, frames=frames)


class PerfectModel:
    """Returns the GT thing instance every frame (drives HOTA ~ 1.0)."""

    key = "perfect"

    def load(self, device) -> None:  # noqa: ARG002
        return None

    def __call__(self, image: np.ndarray) -> FramePrediction:
        h, w, _ = image.shape
        return FramePrediction(
            masks=_rect_mask(h, w, (0, 0, 4, 4))[None],
            categories=torch.tensor([11], dtype=torch.int64),
            scores=torch.tensor([0.9], dtype=torch.float32),
        )


class EmbeddingModel:
    """Returns the GT thing instance with a per-instance embedding + centroid.

    Drives the appearance trackers: every frame the same instance carries the
    same embedding (a unit vector) and a centroid at the mask's pixel mean, so a
    cosine/learned tracker can match it across frames.
    """

    key = "embedding"

    def __init__(self, *, embed_dim: int = 256) -> None:
        self.embed_dim = embed_dim

    def load(self, device) -> None:  # noqa: ARG002
        return None

    def __call__(self, image: np.ndarray) -> FramePrediction:
        h, w, _ = image.shape
        mask = _rect_mask(h, w, (0, 0, 4, 4))
        emb = torch.zeros((1, self.embed_dim), dtype=torch.float32)
        emb[0, 0] = 1.0
        # Centroid of the 4x4 top-left rectangle, ordered (x, y).
        centroid = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
        return FramePrediction(
            masks=mask[None],
            categories=torch.tensor([11], dtype=torch.int64),
            scores=torch.tensor([0.9], dtype=torch.float32),
            embeddings=emb,
            centroids=centroid,
        )


class ScoreSplitDataset:
    """Frames whose GT carries one stable class-11 thing instance."""

    key = "score-split"
    thing_ids = _THING_IDS_SPLIT
    offset = _OFFSET

    def __init__(self, *, n_frames: int = 3) -> None:
        self.n_frames = n_frames

    def sequences(self) -> Iterator[SequenceSample]:
        h = w = 8
        gt = np.zeros((h, w), dtype=np.int64)
        gt[0:4, 0:4] = 11 * _OFFSET + 1  # stable instance across frames
        frames = ((np.zeros((h, w, 3), np.uint8), gt) for _ in range(self.n_frames))
        yield SequenceSample(sequence_id="000000", length=self.n_frames, frames=frames)


class ScoreSplitModel:
    """Emits one above-threshold (0.9) and one below-threshold (0.1) instance.

    The kept instance is a stable class-11 mask (top-left); the suppressed one
    is a non-overlapping class-13 mask (bottom-right) with a low score.
    """

    key = "score-split"

    def load(self, device) -> None:  # noqa: ARG002
        return None

    def __call__(self, image: np.ndarray) -> FramePrediction:
        h, w, _ = image.shape
        kept = _rect_mask(h, w, (0, 0, 4, 4))
        dropped = _rect_mask(h, w, (4, 4, 8, 8))
        return FramePrediction(
            masks=torch.stack([kept, dropped]),
            categories=torch.tensor([11, 13], dtype=torch.int64),
            scores=torch.tensor([0.9, 0.1], dtype=torch.float32),
        )
