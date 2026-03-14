# tests/unitrack/lifecycle/test_standard_lifecycle.py
from __future__ import annotations

import torch
from unitrack.data import FrameContext, MatchOutcome, Tracklets
from unitrack.lifecycle import StandardLifecycle, TrackletStatus


def _t(*, status, hits, tsu):
    n = len(status)
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.tensor([int(s) for s in status], dtype=torch.int8),
        hits=torch.tensor(hits, dtype=torch.int32),
        time_since_update=torch.tensor(tsu, dtype=torch.int32),
        age=torch.tensor(
            [h + t for h, t in zip(hits, tsu, strict=False)], dtype=torch.int32
        ),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )


def test_tentative_promotes_to_active_after_min_hits_consecutive_matches():
    cs = _t(status=[TrackletStatus.Tentative], hits=[1], tsu=[0])
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )
    out = StandardLifecycle(min_hits=2, max_age=3)(cs, match, FrameContext.make(1))
    # hits goes from 1 to 2 — meets threshold.
    assert int(out.status[0]) == int(TrackletStatus.Active)


def test_active_unmatched_increments_tsu_and_eventually_lost():
    cs = _t(status=[TrackletStatus.Active], hits=[5], tsu=[3])
    match = MatchOutcome(
        matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
        tracklets_residual_index=torch.zeros(1, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    out = StandardLifecycle(min_hits=2, max_age=3)(cs, match, FrameContext.make(1))
    # tsu was 3, increments to 4 > max_age=3 → Lost.
    assert int(out.status[0]) == int(TrackletStatus.Lost)
    assert int(out.time_since_update[0]) == 4


def test_tentative_misses_become_removed():
    cs = _t(status=[TrackletStatus.Tentative], hits=[1], tsu=[0])
    match = MatchOutcome(
        matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
        tracklets_residual_index=torch.zeros(1, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    out = StandardLifecycle(min_hits=2, max_age=3)(cs, match, FrameContext.make(1))
    # Tentative tracklets that miss are removed (filtered out).
    assert out.batch_size[0] == 0
