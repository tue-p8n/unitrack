"""Pin SoftLifecycle and StandardLifecycle to the same status/counter logic.

The two policies must produce identical ``status`` / ``hits`` /
``time_since_update`` / ``age`` values for every tracklet that StandardLifecycle
*keeps* (status != Removed). SoftLifecycle keeps the Removed rows too — that
shape-stability is its whole point — but it must not drift on the bookkeeping.
"""

from __future__ import annotations

import torch
from unitrack.data import FrameContext, MatchOutcome, Tracklets
from unitrack.lifecycle import StandardLifecycle
from unitrack.lifecycle.soft import SoftLifecycle


def _make_cs(
    n: int, status: list[int], hits: list[int], tsu: list[int], age: list[int]
) -> Tracklets:
    """Each call returns a fresh Tracklets — both policies mutate via .set() chains,
    so reusing one instance across two policies leaks state between them."""
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.tensor(status, dtype=torch.int8),
        hits=torch.tensor(hits, dtype=torch.int32),
        time_since_update=torch.tensor(tsu, dtype=torch.int32),
        age=torch.tensor(age, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )


def _match(matched_indices: list[int]) -> MatchOutcome:
    if matched_indices:
        pairs = torch.tensor([[i, i] for i in matched_indices], dtype=torch.int64)
    else:
        pairs = torch.zeros((0, 2), dtype=torch.int64)
    return MatchOutcome(
        matched_pairs=pairs,
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(pairs.shape[0]),
        batch_size=[],
    )


def _compare_live(hard_out, soft_out):
    """For every row that StandardLifecycle keeps, the four bookkeeping
    fields must agree with SoftLifecycle's value at the same identity."""
    live_ids = hard_out.id
    soft_keep = torch.isin(soft_out.id, live_ids)
    soft_live = soft_out[soft_keep]
    # Identities must line up (both policies preserve row order on the live subset).
    assert torch.equal(soft_live.id, live_ids)
    assert torch.equal(soft_live.status, hard_out.status)
    assert torch.equal(soft_live.hits, hard_out.hits)
    assert torch.equal(soft_live.time_since_update, hard_out.time_since_update)
    assert torch.equal(soft_live.age, hard_out.age)


def test_promote_tentative_on_match_parity():
    match = _match([0, 1])  # rows 0,1 match; row 2 misses
    hard = StandardLifecycle(min_hits=2, max_age=10)(
        _make_cs(n=3, status=[0, 0, 0], hits=[1, 1, 1], tsu=[0, 0, 0], age=[1, 1, 1]),
        match,
        FrameContext.make(1),
    )
    soft = SoftLifecycle(min_hits=2, max_age=10)(
        _make_cs(n=3, status=[0, 0, 0], hits=[1, 1, 1], tsu=[0, 0, 0], age=[1, 1, 1]),
        match,
        FrameContext.make(1),
    )
    _compare_live(hard, soft)


def test_active_miss_to_lost_parity():
    from unitrack.lifecycle import TrackletStatus

    cfg = {
        "n": 2,
        "status": [int(TrackletStatus.Active), int(TrackletStatus.Active)],
        "hits": [5, 5],
        "tsu": [1, 1],
        "age": [10, 10],
    }
    match = _match([])  # nothing matched
    hard = StandardLifecycle(min_hits=1, max_age=1)(
        _make_cs(**cfg), match, FrameContext.make(2)
    )
    soft = SoftLifecycle(min_hits=1, max_age=1)(
        _make_cs(**cfg), match, FrameContext.make(2)
    )
    _compare_live(hard, soft)


def test_lost_reid_promotion_parity():
    from unitrack.lifecycle import TrackletStatus

    cfg = {
        "n": 2,
        "status": [int(TrackletStatus.Lost), int(TrackletStatus.Lost)],
        "hits": [3, 3],
        "tsu": [2, 2],
        "age": [5, 5],
    }
    match = _match([0])  # row 0 re-matched; row 1 still missing
    hard = StandardLifecycle(min_hits=1, max_age=1, allow_reid=2)(
        _make_cs(**cfg), match, FrameContext.make(3)
    )
    soft = SoftLifecycle(min_hits=1, max_age=1, allow_reid=2)(
        _make_cs(**cfg), match, FrameContext.make(3)
    )
    _compare_live(hard, soft)


def test_tentative_miss_within_grace_parity():
    cfg = {"n": 2, "status": [0, 0], "hits": [1, 1], "tsu": [0, 0], "age": [1, 1]}
    match = _match([])
    hard = StandardLifecycle(min_hits=2, max_age=3, grace_period=2)(
        _make_cs(**cfg), match, FrameContext.make(2)
    )
    soft = SoftLifecycle(min_hits=2, max_age=3, grace_period=2)(
        _make_cs(**cfg), match, FrameContext.make(2)
    )
    _compare_live(hard, soft)


def test_grace_period_boundary_parity():
    """age == grace_period should NOT be removed (the inequality is strict)."""
    cfg = {"n": 2, "status": [0, 0], "hits": [1, 1], "tsu": [0, 0], "age": [1, 1]}
    match = _match([])
    grace = 2
    hard = StandardLifecycle(min_hits=2, max_age=10, grace_period=grace)(
        _make_cs(**cfg), match, FrameContext.make(2)
    )
    soft = SoftLifecycle(min_hits=2, max_age=10, grace_period=grace)(
        _make_cs(**cfg), match, FrameContext.make(2)
    )
    # Both keep tentative rows when age == grace_period.
    assert hard.batch_size[0] == 2
    _compare_live(hard, soft)


def test_soft_lifecycle_keeps_removed_rows():
    """SoftLifecycle must NOT drop rows even when status becomes Removed."""
    from unitrack.lifecycle import TrackletStatus

    cs = _make_cs(n=3, status=[0, 0, 0], hits=[1, 1, 1], tsu=[0, 0, 0], age=[1, 1, 1])
    match = _match([])  # all miss → all marked Removed (grace_period=0)
    soft = SoftLifecycle(min_hits=2, max_age=10, grace_period=0)(
        cs, match, FrameContext.make(2)
    )
    assert soft.batch_size[0] == 3
    assert (soft.status == int(TrackletStatus.Removed)).all()
