# tests/unitrack/lifecycle/test_status.py
from __future__ import annotations

import torch
from unitrack.lifecycle import TrackletStatus


def test_enum_members():
    assert {s.name for s in TrackletStatus} == {
        "Tentative",
        "Active",
        "Lost",
        "Removed",
    }


def test_int_values_unique_and_stable():
    # Persisted values: do not renumber casually.
    assert int(TrackletStatus.Tentative) == 0
    assert int(TrackletStatus.Active) == 1
    assert int(TrackletStatus.Lost) == 2
    assert int(TrackletStatus.Removed) == 3


def test_can_be_packed_into_int8_tensor():
    t = torch.tensor([s.value for s in TrackletStatus], dtype=torch.int8)
    assert t.dtype is torch.int8
    assert t.tolist() == [0, 1, 2, 3]
