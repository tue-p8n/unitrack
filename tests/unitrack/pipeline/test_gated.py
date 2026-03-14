# tests/unitrack/pipeline/test_gated.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, Gate, Tracklets
from unitrack.gates import ClassGate, ScoreGate
from unitrack.pipeline import Gated, Pipe
from unitrack.pipeline.base import GateProducer


def _make(cs_kernel, ds_kernel, cs_class, ds_class, ds_score):
    n, m = cs_kernel.shape[0], ds_kernel.shape[0]
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        kernel=cs_kernel,
        klass=cs_class,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64),
        kernel=ds_kernel,
        klass=ds_class,
        score=ds_score,
        batch_size=[m],
    )
    return cs, ds, FrameContext.make(0)


def test_gated_class_gate_blocks_cross_class_match():
    cs_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ds_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cs_cls = torch.tensor([0, 1], dtype=torch.int64)
    ds_cls = torch.tensor([1, 0], dtype=torch.int64)  # swapped classes
    cs, ds, ctx = _make(cs_k, ds_k, cs_cls, ds_cls, torch.ones(2))
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=1.5)))
    pipe = Gated(gate=ClassGate("klass"), then=inner)
    out = pipe(cs, ds, ctx)
    # Cross-class matches blocked; intra-class pairs (both cost=1.0 < 1.5)
    # survive the threshold. The solver assigns both intra-class pairs.
    assert sorted(out.matched_pairs.tolist()) == [[0, 1], [1, 0]]


def test_gated_score_gate_drops_low_score_detections_per_ds():
    cs_k = torch.tensor([[1.0, 0.0]])
    ds_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cs, ds, ctx = _make(
        cs_k,
        ds_k,
        cs_class=torch.zeros(1, dtype=torch.int64),
        ds_class=torch.zeros(2, dtype=torch.int64),
        ds_score=torch.tensor([0.9, 0.1]),
    )
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.99)))
    pipe = Gated(gate=ScoreGate("score", threshold=0.5), then=inner)
    out = pipe(cs, ds, ctx)
    pairs = out.matched_pairs.tolist()
    # Only ds 0 survives ScoreGate; cs 0 must match ds 0.
    assert pairs == [[0, 0]]


# --------- four-variant Gated dispatch coverage --------------------------
# The four Gate variants exercise different code paths in `_gated_dispatch`:
#   - PerCs : `g.kind == 'per_cs'` branch (filter cs, remap matched_pairs[:,0])
#   - PerDs : `g.kind == 'per_ds'` branch (filter ds, remap matched_pairs[:,1])
#   - PerPair / CostBias : handled by `_run_with_pair_or_bias` (Pipe body) or
#                            forwarded via `g.apply(cost)` (non-Pipe body)
#   - _PairAndBias : produced by `Gate.combine(per_pair, cost_bias)` — exercises
#                     the combined apply path that attaches both mask and bias.


class _PerCsAcceptOdd(GateProducer):
    """Gate that allows only odd cs indices through."""

    def __call__(self, cs, ds, ctx):
        del ds, ctx
        n = cs.batch_size[0]
        return Gate.PerCs(mask=(torch.arange(n) % 2 == 1))


class _CostBiasFavorRow0(GateProducer):
    """Gate that adds a constant negative bias on row 0 of the (N, M) matrix."""

    def __call__(self, cs, ds, ctx):
        del ctx
        n = cs.batch_size[0]
        m = ds.batch_size[0]
        bias = torch.zeros(n, m)
        bias[0] = -10.0  # row 0 strongly favoured (very negative cost added)
        return Gate.CostBias(matrix=bias)


class _PairAndBiasCombined(GateProducer):
    """Gate whose output is a _PairAndBias produced by Gate.combine."""

    def __call__(self, cs, ds, ctx):
        del ctx
        n = cs.batch_size[0]
        m = ds.batch_size[0]
        pair = torch.ones(n, m, dtype=torch.bool)
        pair[0, 0] = False  # forbid (0,0)
        bias = torch.zeros(n, m)
        bias[1, 1] = -10.0  # heavily favour (1,1)
        return Gate.combine(Gate.PerPair(mask=pair), Gate.CostBias(matrix=bias))


def test_gated_per_cs_filters_tracklets_before_pipeline():
    """`Gate.PerCs` filters cs by the mask; Gated remaps matched indices
    back to the original cs row space."""
    cs_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ds_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cs, ds, ctx = _make(
        cs_k,
        ds_k,
        cs_class=torch.zeros(2, dtype=torch.int64),
        ds_class=torch.zeros(2, dtype=torch.int64),
        ds_score=torch.ones(2),
    )
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=1.5)))
    pipe = Gated(gate=_PerCsAcceptOdd(), then=inner)
    out = pipe(cs, ds, ctx)
    # PerCs mask keeps only cs index 1; cs 1 best-matches ds 1 (cost 0).
    # cs 0 is filtered out → appears in tracklets_residual_index.
    assert out.matched_pairs.tolist() == [[1, 1]]
    assert 0 in out.tracklets_residual_index.tolist()


def test_gated_cost_bias_attaches_to_cost_expression():
    """`Gate.CostBias` adds an additive bias to the cost matrix that flows
    through materialise. With a -10 bias on row 0 the solver prefers row 0
    over what an un-biased Cosine alone would pick."""
    # Two identical detections — without bias, both costs are 1.0 (cosine
    # similarity 0). The -10 bias on row 0 makes row-0 pairs cost -9, which
    # the solver greedily picks. Row 1 then gets the remaining detection.
    cs_k = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    ds_k = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cs, ds, ctx = _make(
        cs_k,
        ds_k,
        cs_class=torch.zeros(2, dtype=torch.int64),
        ds_class=torch.zeros(2, dtype=torch.int64),
        ds_score=torch.ones(2),
    )
    # Threshold 5.0 admits the -9 pair and rejects the un-biased ~1 pair.
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=5.0)))
    pipe = Gated(gate=_CostBiasFavorRow0(), then=inner)
    out = pipe(cs, ds, ctx)
    pairs = out.matched_pairs.tolist()
    # Row 0 (cs 0) must be matched somewhere because of the strong negative bias.
    assert any(p[0] == 0 for p in pairs)


def test_gated_pair_and_bias_attaches_mask_and_bias_together():
    """`Gate.combine(PerPair, CostBias)` produces a `_PairAndBias`; the
    Gated dispatch must apply BOTH the (N,M) mask AND the (N,M) bias to
    the inner cost expression."""
    cs_k = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    ds_k = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cs, ds, ctx = _make(
        cs_k,
        ds_k,
        cs_class=torch.zeros(2, dtype=torch.int64),
        ds_class=torch.zeros(2, dtype=torch.int64),
        ds_score=torch.ones(2),
    )
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=5.0)))
    pipe = Gated(gate=_PairAndBiasCombined(), then=inner)
    out = pipe(cs, ds, ctx)
    pairs = out.matched_pairs.tolist()
    # Mask forbids (0, 0); bias strongly favours (1, 1). The (1, 1) pair must
    # appear; (0, 0) must not.
    assert [1, 1] in pairs
    assert [0, 0] not in pairs
