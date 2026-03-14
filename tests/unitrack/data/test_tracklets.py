# tests/unitrack/data/test_tracklets.py
from __future__ import annotations

import torch
from unitrack.data import Tracklets
from unitrack.lifecycle import TrackletStatus


def test_empty_constructor_zero_rows():
    t = Tracklets.empty()
    assert t.id.shape == (0,)
    assert t.id.dtype is torch.int64
    assert t.status.shape == (0,)
    assert t.status.dtype is torch.int8
    assert t.batch_size == torch.Size((0,))


def test_extra_user_fields_are_carried_through():
    t = Tracklets(
        id=torch.tensor([1, 2, 3], dtype=torch.int64),
        status=torch.tensor([TrackletStatus.Active] * 3, dtype=torch.int8),
        hits=torch.tensor([5, 5, 5], dtype=torch.int32),
        time_since_update=torch.zeros(3, dtype=torch.int32),
        age=torch.tensor([10, 11, 12], dtype=torch.int32),
        frame_started=torch.tensor([0, 0, 0], dtype=torch.int32),
        frame_last_seen=torch.tensor([10, 11, 12], dtype=torch.int32),
        kernel=torch.randn(3, 8),
        batch_size=[3],
    )
    assert t.kernel.shape == (3, 8)
    assert t.id.tolist() == [1, 2, 3]
    assert (t.status == TrackletStatus.Active).all()


def test_indexing_preserves_user_fields():
    t = Tracklets(
        id=torch.tensor([1, 2, 3], dtype=torch.int64),
        status=torch.tensor(
            [TrackletStatus.Active, TrackletStatus.Lost, TrackletStatus.Tentative],
            dtype=torch.int8,
        ),
        hits=torch.tensor([1, 2, 3], dtype=torch.int32),
        time_since_update=torch.tensor([0, 1, 0], dtype=torch.int32),
        age=torch.tensor([5, 6, 7], dtype=torch.int32),
        frame_started=torch.tensor([0, 0, 0], dtype=torch.int32),
        frame_last_seen=torch.tensor([5, 5, 7], dtype=torch.int32),
        kernel=torch.arange(15.0).reshape(3, 5),
        batch_size=[3],
    )
    sub = t[torch.tensor([0, 2])]
    assert sub.batch_size == torch.Size((2,))
    assert sub.id.tolist() == [1, 3]
    assert sub.kernel.shape == (2, 5)
