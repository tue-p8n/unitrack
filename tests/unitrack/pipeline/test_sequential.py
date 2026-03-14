# tests/unitrack/pipeline/test_sequential.py
from __future__ import annotations

import pytest
import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    Gate,
    MatchOutcome,
    Tracklets,
)
from unitrack.pipeline import Pipe, Sequential
from unitrack.pipeline.base import PipelineTypeError


def _cs(kernel: torch.Tensor) -> Tracklets:
    n = kernel.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        kernel=kernel,
        batch_size=[n],
    )


def _ds(kernel: torch.Tensor) -> Detections:
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_sequential_match_outcome_cascades_residuals():
    # Stage 1 only matches a tiny threshold; stage 2 picks up the rest.
    cs = _cs(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    ds = _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    pipeline = Sequential[MatchOutcome](
        [
            Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.0001))),
            Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        ]
    )
    out = pipeline(cs, ds, FrameContext.make(0))
    pairs = sorted(map(tuple, out.matched_pairs.tolist()))
    assert pairs == [(0, 0), (1, 1)]


def test_sequential_gate_folds_children_via_combine():
    class _G:
        def __init__(self, mask):
            self.mask = mask

        def __call__(self, _cs, _ds, _ctx):
            return Gate.PerPair(mask=self.mask)

    a_mask = torch.tensor([[True, True], [True, False]])
    b_mask = torch.tensor([[True, False], [True, True]])
    seq = Sequential[Gate]([_G(a_mask), _G(b_mask)])
    out = seq(_cs(torch.zeros(2, 2)), _ds(torch.zeros(2, 2)), FrameContext.make(0))
    expected = a_mask & b_mask
    assert torch.equal(out.mask, expected)


def test_sequential_cost_expression_rejected_at_construction():
    with pytest.raises(PipelineTypeError, match="CostExpression"):
        Sequential[CostExpression]([Cosine("kernel"), Cosine("kernel")])


def test_sequential_mixed_t_rejected():
    class _C:
        def __call__(self, _cs, _ds, _ctx):
            return CostExpression.from_matrix(torch.zeros(0, 0))

    class _M:
        def __call__(self, _cs, _ds, _ctx, cost=None):  # noqa: ARG002
            return MatchOutcome.empty()

    with pytest.raises(PipelineTypeError, match=r"(?i)same"):
        Sequential[MatchOutcome]([_M(), _C()])


def test_sequential_gate_rejects_match_typed_children():
    """``Sequential[Gate]`` must refuse Associator children at construction.

    Before the subscript bound the expected kind, ``Sequential[Gate]([pipe])``
    silently classified the Pipe as ``match`` and ran a match cascade against
    the user's typed expectation.
    """
    pipe = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    with pytest.raises(PipelineTypeError, match="Sequential\\[Gate\\]"):
        Sequential[Gate]([pipe, pipe])


def test_sequential_match_rejects_gate_typed_children():
    """``Sequential[MatchOutcome]`` must refuse Gate children at construction."""

    class _G:
        def __call__(self, _cs, _ds, _ctx):
            return Gate.PerPair(mask=torch.zeros(0, 0, dtype=torch.bool))

    with pytest.raises(PipelineTypeError, match="Sequential\\[MatchOutcome\\]"):
        Sequential[MatchOutcome]([_G(), _G()])


def test_sequential_match_breaks_when_cs_empties_mid_cascade():
    """When stage 1 consumes all tracklets, stage 2 must short-circuit.
    The final ``tracklets_residual_index`` should be empty (in the original
    cs space) and ``detections_residual_index`` should hold every unmatched
    detection."""
    cs = _cs(torch.tensor([[1.0, 0.0]]))  # one tracklet
    ds = _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))  # two detections
    pipeline = Sequential[MatchOutcome](
        [
            # Stage 1: matches cs 0 to ds 0 with cost ~0, consuming the only cs.
            Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
            # Stage 2: would try to match again but cs is empty -> early break.
            Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        ]
    )
    out = pipeline(cs, ds, FrameContext.make(0))
    assert out.matched_pairs.tolist() == [[0, 0]]
    assert out.tracklets_residual_index.numel() == 0
    # ds 1 should be reported unmatched, lifted back to its original index.
    assert out.detections_residual_index.tolist() == [1]


def test_sequential_match_breaks_when_ds_empties_mid_cascade():
    """Symmetric: when stage 1 consumes all detections, stage 2 short-circuits."""
    cs = _cs(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))  # two tracklets
    ds = _ds(torch.tensor([[1.0, 0.0]]))  # one detection
    pipeline = Sequential[MatchOutcome](
        [
            Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
            Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        ]
    )
    out = pipeline(cs, ds, FrameContext.make(0))
    assert out.matched_pairs.tolist() == [[0, 0]]
    # cs 1 unmatched, lifted back to original index space.
    assert out.tracklets_residual_index.tolist() == [1]
    assert out.detections_residual_index.numel() == 0
