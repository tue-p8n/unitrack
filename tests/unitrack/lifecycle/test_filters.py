# tests/unitrack/lifecycle/test_filters.py
from __future__ import annotations

import torch
from unitrack.data import Tracklets
from unitrack.lifecycle import MaxAgeFilter, TrackletStatus


def _t(*, time_since_update: list[int]) -> Tracklets:
    n = len(time_since_update)
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.full((n,), int(TrackletStatus.Active), dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.tensor(time_since_update, dtype=torch.int32),
        age=torch.tensor(time_since_update, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )


def test_max_age_filter_keeps_recent_tracklets():
    cs = _t(time_since_update=[0, 1, 4, 5])
    keep = MaxAgeFilter(max_age=4)(cs)
    assert keep.tolist() == [True, True, True, False]
