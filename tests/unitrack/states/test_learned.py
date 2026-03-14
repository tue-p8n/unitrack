from __future__ import annotations

import torch
from torch import nn
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states import LearnedObservation, LearnedProcess


def _t(emb: torch.Tensor) -> Tracklets:
    n = emb.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        emb=emb,
        batch_size=[n],
    )


def _match(pairs) -> MatchOutcome:
    return MatchOutcome(
        matched_pairs=torch.tensor(pairs, dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(len(pairs)),
        batch_size=[],
    )


def test_process_applies_module_with_dt():
    proc = LearnedProcess("emb", lambda v, dt: v * dt)
    out = proc(_t(torch.ones(3, 4)), FrameContext.make(0, delta=2.0))
    assert torch.allclose(out.emb, torch.full((3, 4), 2.0))


def test_process_noop_on_empty_snapshot():
    proc = LearnedProcess("emb", lambda v, _: v + 1)
    out = proc(_t(torch.zeros(0, 4)), FrameContext.make(0, delta=1.0))
    assert out.emb.shape == (0, 4)


def test_observation_updates_only_matched():
    obs = LearnedObservation("emb", "emb", lambda _, m: m)  # adopt measurement
    cs = _t(torch.zeros(2, 3))
    ds = Detections(
        index=torch.arange(1), emb=torch.tensor([[1.0, 1.0, 1.0]]), batch_size=[1]
    )
    out = obs(cs, ds, _match([[0, 0]]), FrameContext.make(0))
    assert torch.allclose(out.emb[0], torch.ones(3))
    assert torch.allclose(out.emb[1], torch.zeros(3))  # unmatched untouched


def test_gradient_flows_to_module_parameters():
    gru = nn.GRUCell(4, 4)
    obs = LearnedObservation("emb", "emb", lambda t, m: gru(m, t))
    cs = _t(torch.randn(2, 4))
    ds = Detections(index=torch.arange(2), emb=torch.randn(2, 4), batch_size=[2])
    out = obs(cs, ds, _match([[0, 0], [1, 1]]), FrameContext.make(0))
    out.emb.pow(2).sum().backward()
    grad = sum(p.grad.norm().item() for p in gru.parameters() if p.grad is not None)
    assert grad > 0.0
