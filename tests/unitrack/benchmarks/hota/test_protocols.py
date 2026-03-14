from typing import ClassVar

import torch
from unitrack.benchmarks.hota.protocols import DatasetAdapter, ModelAdapter
from unitrack.benchmarks.hota.types import FramePrediction


class _M:
    key = "m"

    def load(self, device): ...

    def __call__(self, image):  # noqa: ARG002
        return FramePrediction(
            masks=torch.zeros((0, 4, 4), dtype=torch.bool),
            categories=torch.zeros((0,), dtype=torch.int64),
            scores=torch.zeros((0,), dtype=torch.float32),
        )


class _D:
    key = "d"
    thing_ids: ClassVar[list[int]] = [11]
    offset = 1000

    def sequences(self):
        return iter(())


def test_adapters_are_structural():
    assert isinstance(_M(), ModelAdapter)
    assert isinstance(_D(), DatasetAdapter)
