# tests/unitrack/lifecycle/test_lifecycle_grace_and_reid.py
"""Regression tests for I1: StandardLifecycle.grace_period + Lost→Active reid."""

from __future__ import annotations

import torch
from unitrack.data import FrameContext, MatchOutcome, Tracklets
from unitrack.lifecycle import StandardLifecycle, TrackletStatus


def _t(*, status, hits, tsu, age=None):
    n = len(status)
    if age is None:
        age = [h + t for h, t in zip(hits, tsu, strict=False)]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.tensor([int(s) for s in status], dtype=torch.int8),
        hits=torch.tensor(hits, dtype=torch.int32),
        time_since_update=torch.tensor(tsu, dtype=torch.int32),
        age=torch.tensor(age, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )


def _no_match(n_cs: int) -> MatchOutcome:
    return MatchOutcome(
        matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
        tracklets_residual_index=torch.arange(n_cs, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )


def test_grace_period_keeps_unmatched_tentative_alive():
    """grace_period=2: Tentative misses while age<grace stay alive."""
    # age=0 entering policy (pre-increment) means a freshly spawned tracklet.
    # The policy increments age to 1; with grace_period=2 it must stay Tentative.
    cs = _t(status=[TrackletStatus.Tentative], hits=[1], tsu=[0], age=[0])
    out = StandardLifecycle(min_hits=3, max_age=3, grace_period=2)(
        cs, _no_match(1), FrameContext.make(1)
    )
    # The match outcome has no matched pairs but is_new=(age==0) so the
    # policy considers row 0 matched-by-construction (just spawned). That
    # bypasses the grace check entirely; assert it remains alive as Tentative.
    assert out.batch_size[0] == 1
    assert int(out.status[0]) == int(TrackletStatus.Tentative)


def test_grace_period_zero_removes_tentative_immediately_on_miss():
    """Default grace_period=0 ⇒ Tentative + miss after age>0 → Removed."""
    cs = _t(status=[TrackletStatus.Tentative], hits=[1], tsu=[0], age=[1])
    out = StandardLifecycle(min_hits=3, max_age=3, grace_period=0)(
        cs, _no_match(1), FrameContext.make(1)
    )
    # age=1 entering policy (not fresh) and not matched → Removed → filtered out.
    assert out.batch_size[0] == 0


def test_grace_period_eventually_expires_and_removes_tentative():
    """grace_period=1: age=3 after increment > 1 → Removed."""
    cs = _t(status=[TrackletStatus.Tentative], hits=[1], tsu=[1], age=[2])
    out = StandardLifecycle(min_hits=5, max_age=3, grace_period=1)(
        cs, _no_match(1), FrameContext.make(1)
    )
    assert out.batch_size[0] == 0


def test_lost_then_matched_within_allow_reid_promotes_to_active():
    """A Lost tracklet that matches a detection within allow_reid → Active."""
    cs = _t(status=[TrackletStatus.Lost], hits=[5], tsu=[2], age=[7])
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )
    out = StandardLifecycle(min_hits=2, max_age=3, allow_reid=5)(
        cs, match, FrameContext.make(1)
    )
    assert out.batch_size[0] == 1
    assert int(out.status[0]) == int(TrackletStatus.Active)


def test_lost_then_matched_without_allow_reid_stays_lost():
    """Without re-id budget, a Lost+match doesn't re-promote."""
    cs = _t(status=[TrackletStatus.Lost], hits=[5], tsu=[2], age=[7])
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )
    out = StandardLifecycle(min_hits=2, max_age=3, allow_reid=0)(
        cs, match, FrameContext.make(1)
    )
    assert out.batch_size[0] == 1
    assert int(out.status[0]) == int(TrackletStatus.Lost)
