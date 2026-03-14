# tests/unitrack/assignment/test_associate.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    Tracklets,
)


def _empty_cs(n: int) -> Tracklets:
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )


def _empty_ds(m: int) -> Detections:
    return Detections(index=torch.arange(m, dtype=torch.int64), batch_size=[m])


def test_associate_identity_matrix():
    n, m = 3, 3
    matrix = torch.eye(3) * 0.0 + torch.full((3, 3), 0.9)
    matrix.fill_diagonal_(0.1)
    expr = CostExpression.from_matrix(matrix)
    out = Associate(Jonker(threshold=0.5))(
        _empty_cs(n), _empty_ds(m), FrameContext.make(0), expr
    )
    pairs = sorted(map(tuple, out.matched_pairs.tolist()))
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_associate_threshold_drops_high_cost_pairs():
    expr = CostExpression.from_matrix(torch.tensor([[0.1, 0.9], [0.9, 0.1]]))
    out = Associate(Jonker(threshold=0.5))(
        _empty_cs(2), _empty_ds(2), FrameContext.make(0), expr
    )
    pairs = sorted(map(tuple, out.matched_pairs.tolist()))
    assert pairs == [(0, 0), (1, 1)]
    expr = CostExpression.from_matrix(torch.tensor([[0.1, 0.9], [0.9, 0.1]]))
    out_strict = Associate(Jonker(threshold=0.05))(
        _empty_cs(2),
        _empty_ds(2),
        FrameContext.make(0),
        expr,
    )
    assert out_strict.matched_pairs.shape[0] == 0


def test_residuals_are_indices_into_original_inputs():
    expr = CostExpression.from_matrix(torch.tensor([[0.1, 0.9, 0.9], [0.9, 0.9, 0.9]]))
    out = Associate(Jonker(threshold=0.5))(
        _empty_cs(2), _empty_ds(3), FrameContext.make(0), expr
    )
    assert out.matched_pairs.tolist() == [[0, 0]]
    assert out.tracklets_residual_index.tolist() == [1]
    assert sorted(out.detections_residual_index.tolist()) == [1, 2]
