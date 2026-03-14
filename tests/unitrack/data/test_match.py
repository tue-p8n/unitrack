# tests/unitrack/data/test_match.py
from __future__ import annotations

import pytest
import torch
from unitrack.data import MatchOutcome


def test_empty_match():
    m = MatchOutcome.empty()
    assert m.matched_pairs.shape == (0, 2)
    assert m.tracklets_residual_index.shape == (0,)
    assert m.detections_residual_index.shape == (0,)
    assert m.per_match_cost.shape == (0,)


def test_concrete_match():
    m = MatchOutcome(
        matched_pairs=torch.tensor([[0, 1], [2, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.tensor([1, 3], dtype=torch.int64),
        detections_residual_index=torch.tensor([2], dtype=torch.int64),
        per_match_cost=torch.tensor([0.1, 0.4]),
        batch_size=[],
    )
    assert m.matched_pairs.tolist() == [[0, 1], [2, 0]]
    assert m.tracklets_residual_index.tolist() == [1, 3]
    assert m.detections_residual_index.tolist() == [2]
    assert m.per_match_cost.tolist() == pytest.approx([0.1, 0.4], rel=1e-5)
