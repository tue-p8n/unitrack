# tests/unitrack/lifecycle/test_visibility.py
from __future__ import annotations

import torch
from unitrack.data import MatchOutcome, Tracklets
from unitrack.lifecycle import (
    ConfirmedOnly,
    IncludeTentative,
    TrackletStatus,
)


def _t(statuses):
    n = len(statuses)
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64) + 10,
        status=torch.tensor([int(s) for s in statuses], dtype=torch.int8),
        hits=torch.zeros(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.zeros(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )


def test_confirmed_only_returns_only_active_matched_ids():
    cs = _t([TrackletStatus.Tentative, TrackletStatus.Active, TrackletStatus.Lost])
    # Only the Active row was matched this frame.
    match = MatchOutcome(
        matched_pairs=torch.tensor([[1, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.tensor([0, 2], dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )
    out = ConfirmedOnly()(cs, match)
    assert out.tolist() == [11]


def test_include_tentative_also_returns_tentative_ids():
    cs = _t([TrackletStatus.Tentative, TrackletStatus.Active])
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 0], [1, 1]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(2),
        batch_size=[],
    )
    out = IncludeTentative()(cs, match)
    assert sorted(out.tolist()) == [10, 11]
