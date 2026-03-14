# tests/unitrack/tracker/test_memory.py
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


def test_initial_state_empty_snapshot_and_next_id_one():
    mem = TrackletMemory(_empty())
    assert mem.snapshot.batch_size[0] == 0
    assert mem.next_id == 1


def test_load_replaces_snapshot_and_next_id():
    mem = TrackletMemory(_empty())
    new_snap = Tracklets(
        id=torch.tensor([7, 8], dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.zeros(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        batch_size=[2],
    )
    mem.load(new_snap, next_id=9)
    assert mem.snapshot.batch_size[0] == 2
    assert mem.next_id == 9


def test_reset_after_load():
    mem = TrackletMemory(_empty())
    new_snap = Tracklets(
        id=torch.tensor([7], dtype=torch.int64),
        status=torch.ones(1, dtype=torch.int8),
        hits=torch.ones(1, dtype=torch.int32),
        time_since_update=torch.zeros(1, dtype=torch.int32),
        age=torch.zeros(1, dtype=torch.int32),
        frame_started=torch.zeros(1, dtype=torch.int32),
        frame_last_seen=torch.zeros(1, dtype=torch.int32),
        batch_size=[1],
    )
    mem.load(new_snap, next_id=9)
    mem.reset()
    assert mem.snapshot.batch_size[0] == 0
    assert mem.next_id == 1


def test_fork_yields_independent_state():
    mem = TrackletMemory(_empty())
    fork = mem.fork()
    snap = Tracklets(
        id=torch.tensor([1], dtype=torch.int64),
        status=torch.ones(1, dtype=torch.int8),
        hits=torch.ones(1, dtype=torch.int32),
        time_since_update=torch.zeros(1, dtype=torch.int32),
        age=torch.zeros(1, dtype=torch.int32),
        frame_started=torch.zeros(1, dtype=torch.int32),
        frame_last_seen=torch.zeros(1, dtype=torch.int32),
        batch_size=[1],
    )
    fork.load(snap, next_id=2)
    assert mem.snapshot.batch_size[0] == 0
    assert fork.snapshot.batch_size[0] == 1
