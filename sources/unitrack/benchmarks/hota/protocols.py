"""Structural protocols for the three harness plug points."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from .types import FramePrediction, SequenceSample

if TYPE_CHECKING:
    from unitrack import MultiStream


@runtime_checkable
class ModelAdapter(Protocol):
    """A HuggingFace model that emits one ``FramePrediction`` per image."""

    key: str

    def load(self, device: torch.device) -> None:
        """Materialize weights/processor on ``device`` (call once before use)."""
        ...

    def __call__(self, image: np.ndarray) -> FramePrediction:
        """Run inference on a single ``(H, W, 3)`` uint8 RGB frame."""
        ...


@runtime_checkable
class DatasetAdapter(Protocol):
    """A source of ground-truth sequences plus its panoptic label space."""

    key: str
    thing_ids: list[int]
    offset: int

    def sequences(self) -> Iterator[SequenceSample]:
        """Yield each ground-truth sequence in deterministic order."""
        ...


@runtime_checkable
class TrackerFactory(Protocol):
    """A factory returning a fresh ``unitrack.MultiStream`` for one sequence."""

    def __call__(self, height: int, width: int) -> MultiStream:
        """Build a fresh tracker sized to a sequence's frame resolution."""
        ...
