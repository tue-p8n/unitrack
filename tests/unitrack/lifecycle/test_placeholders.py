# tests/unitrack/lifecycle/test_placeholders.py
from __future__ import annotations

import torch
from unitrack.data import FrameContext, MatchOutcome, Tracklets
from unitrack.lifecycle import IncludeAll, NoLifecycle, TrackletStatus


def test_no_lifecycle_is_identity_on_snapshot():
    cs = Tracklets(
        id=torch.tensor([1, 2], dtype=torch.int64),
        status=torch.tensor(
            [TrackletStatus.Tentative, TrackletStatus.Active], dtype=torch.int8
        ),
        hits=torch.tensor([1, 5], dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.tensor([1, 5], dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.tensor([3, 5], dtype=torch.int32),
        batch_size=[2],
    )
    match = MatchOutcome.empty()
    out = NoLifecycle()(cs, match, FrameContext.make(6))
    assert torch.equal(out.id, cs.id)
    assert torch.equal(out.status, cs.status)


def test_include_all_returns_full_id_array():
    cs = Tracklets(
        id=torch.tensor([10, 11, 12], dtype=torch.int64),
        status=torch.tensor([TrackletStatus.Active] * 3, dtype=torch.int8),
        hits=torch.zeros(3, dtype=torch.int32),
        time_since_update=torch.zeros(3, dtype=torch.int32),
        age=torch.zeros(3, dtype=torch.int32),
        frame_started=torch.zeros(3, dtype=torch.int32),
        frame_last_seen=torch.zeros(3, dtype=torch.int32),
        batch_size=[3],
    )
    out = IncludeAll()(cs, MatchOutcome.empty())
    assert out.tolist() == [10, 11, 12]
