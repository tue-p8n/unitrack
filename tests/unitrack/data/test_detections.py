# tests/unitrack/data/test_detections.py
from __future__ import annotations

import torch
from unitrack.data import Detections


def test_minimum_required_fields():
    d = Detections(
        index=torch.arange(4, dtype=torch.int64),
        kernel=torch.randn(4, 8),
        batch_size=[4],
    )
    assert d.batch_size == torch.Size((4,))
    assert d.index.dtype is torch.int64
    assert d.kernel.shape == (4, 8)


def test_empty_factory():
    d = Detections.empty()
    assert d.batch_size == torch.Size((0,))
    assert d.index.shape == (0,)


def test_indexing():
    d = Detections(
        index=torch.arange(4, dtype=torch.int64),
        kernel=torch.arange(20.0).reshape(4, 5),
        batch_size=[4],
    )
    sub = d[torch.tensor([1, 3])]
    assert sub.index.tolist() == [1, 3]
    assert sub.kernel.shape == (2, 5)
