from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states import (
    VonMisesFisherDecay,
    VonMisesFisherUpdate,
    vmf_state_entries,
)


def _t(mu: torch.Tensor, kappa: torch.Tensor) -> Tracklets:
    n = mu.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        emb=mu,
        emb_kappa=kappa,
        batch_size=[n],
    )


def _match_one() -> MatchOutcome:
    return MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )


def test_update_keeps_unit_norm_and_raises_kappa():
    mu = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), dim=-1)
    cs = _t(mu.clone(), torch.tensor([5.0]))
    obs = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), dim=-1)
    upd = VonMisesFisherUpdate("emb", "emb", kappa_obs=20.0)
    ds = Detections(index=torch.tensor([0]), emb=obs, batch_size=[1])
    out = upd(cs, ds, _match_one(), FrameContext.make(0))
    assert torch.allclose(out.emb[0].norm(), torch.tensor(1.0), atol=1e-5)
    # resultant length of 5*mu + 20*obs (orthogonal) is sqrt(25 + 400).
    assert torch.allclose(
        out.emb_kappa[0], torch.tensor((25.0 + 400.0) ** 0.5), atol=1e-4
    )


def test_repeated_consistent_obs_converges_to_observation():
    mu = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    obs = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    cs = _t(mu.clone(), torch.tensor([3.0]))
    upd = VonMisesFisherUpdate("emb", "emb", kappa_obs=15.0)
    ds = Detections(index=torch.tensor([0]), emb=obs, batch_size=[1])
    for _ in range(8):
        cs = upd(cs, ds, _match_one(), FrameContext.make(0))
    cos = (cs.emb[0] @ obs[0]).item()
    assert cos > 0.99
    assert cs.emb_kappa[0] > 100.0


def test_decay_lowers_kappa_toward_floor():
    cs = _t(torch.zeros(1, 4), torch.tensor([100.0]))
    proc = VonMisesFisherDecay("emb", tau=1.0, kappa_min=2.0)
    out = proc(cs, FrameContext.make(0, delta=1.0))
    assert torch.allclose(
        out.emb_kappa[0], torch.tensor(100.0 * torch.e**-1), atol=1e-3
    )
    # decay never drops below the floor
    floored = proc(
        _t(torch.zeros(1, 4), torch.tensor([2.5])), FrameContext.make(0, delta=50.0)
    )
    assert torch.allclose(floored.emb_kappa[0], torch.tensor(2.0))


def test_state_entries_keys_and_init_normalizes():
    entries = vmf_state_entries("emb", dim=4, init_kappa=12.0)
    assert set(entries) == {"emb", "emb_kappa"}
    ds = Detections(
        index=torch.tensor([0]),
        emb=torch.tensor([[3.0, 4.0, 0.0, 0.0]]),
        batch_size=[1],
    )
    mu0 = entries["emb"].init(ds, FrameContext.make(0))
    assert torch.allclose(mu0[0].norm(), torch.tensor(1.0), atol=1e-6)
    k0 = entries["emb_kappa"].init(ds, FrameContext.make(0))
    assert torch.allclose(k0, torch.tensor([12.0]))
