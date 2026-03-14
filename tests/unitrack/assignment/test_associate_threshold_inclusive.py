# tests/unitrack/assignment/test_associate_threshold_inclusive.py
"""Regression for I7: Associate post-filter keeps pairs cost == threshold."""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    Tracklets,
)


def _cs_ds(n, m):
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )
    ds = Detections(index=torch.arange(m, dtype=torch.int64), batch_size=[m])
    return cs, ds, FrameContext.make(0)


def test_pair_at_threshold_is_retained_inclusively():
    """Pair cost exactly equal to threshold is kept (spec §5.7: drop above)."""
    cs, ds, ctx = _cs_ds(1, 1)
    cost = CostExpression.from_matrix(torch.tensor([[0.5]]))
    assoc = Associate(Jonker(threshold=0.5))
    out = assoc(cs, ds, ctx, cost)
    assert out.matched_pairs.shape[0] == 1
    assert int(out.matched_pairs[0, 0]) == 0
    assert int(out.matched_pairs[0, 1]) == 0


def test_pair_above_threshold_is_rejected():
    """Pair with cost strictly above threshold is dropped to residuals."""
    cs, ds, ctx = _cs_ds(1, 1)
    cost = CostExpression.from_matrix(torch.tensor([[0.6]]))
    assoc = Associate(Jonker(threshold=0.5))
    out = assoc(cs, ds, ctx, cost)
    assert out.matched_pairs.shape[0] == 0
    assert out.tracklets_residual_index.tolist() == [0]
    assert out.detections_residual_index.tolist() == [0]
