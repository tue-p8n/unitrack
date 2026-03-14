# tests/unitrack/tracker/test_memory_fork.py
"""Regression test for TrackletMemory.fork(): copies the current snapshot + next_id."""

from __future__ import annotations

import torch
from unitrack.data import Tracklets
from unitrack.tracker import TrackletMemory


def _empty(device=None) -> Tracklets:
    z = lambda dt: torch.zeros(0, dtype=dt, device=device)  # noqa: E731
    return Tracklets(
        id=z(torch.int64),
        status=z(torch.int8),
        hits=z(torch.int32),
        time_since_update=z(torch.int32),
        age=z(torch.int32),
        frame_started=z(torch.int32),
        frame_last_seen=z(torch.int32),
        batch_size=[0],
    )


def _one(id_val: int) -> Tracklets:
    return Tracklets(
        id=torch.tensor([id_val], dtype=torch.int64),
        status=torch.ones(1, dtype=torch.int8),
        hits=torch.ones(1, dtype=torch.int32),
        time_since_update=torch.zeros(1, dtype=torch.int32),
        age=torch.zeros(1, dtype=torch.int32),
        frame_started=torch.zeros(1, dtype=torch.int32),
        frame_last_seen=torch.zeros(1, dtype=torch.int32),
        batch_size=[1],
    )


def test_fork_preserves_current_snapshot_and_next_id():
    """Spec §9.1: fork is a cheap copy of the current state, not a reset."""
    mem = TrackletMemory(_empty())
    mem.load(_one(7), next_id=8)
    fork = mem.fork()
    assert fork.snapshot.batch_size[0] == 1
    assert fork.next_id == 8
    assert int(fork.snapshot.id[0]) == 7


def test_fork_state_is_independent_of_parent():
    """Mutating fork.snapshot after fork must not affect parent."""
    mem = TrackletMemory(_empty())
    mem.load(_one(3), next_id=4)
    fork = mem.fork()
    fork.load(_one(99), next_id=100)
    assert int(mem.snapshot.id[0]) == 3
    assert mem.next_id == 4
    assert int(fork.snapshot.id[0]) == 99
    assert fork.next_id == 100
