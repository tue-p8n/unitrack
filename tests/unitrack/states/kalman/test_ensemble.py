from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states.kalman import (
    EnsembleInitializer,
    EnsembleProcess,
    EnsembleUpdate,
    enkf_state_entries,
)


def _t(mean: torch.Tensor, members: torch.Tensor) -> Tracklets:
    n = mean.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        emb=mean,
        emb_ensemble=members,
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


def _spawn(z: torch.Tensor, e: int, d: int, *, std=0.5, seed=0) -> torch.Tensor:
    ds = Detections(index=torch.tensor([0]), emb=z, batch_size=[1])
    return EnsembleInitializer("emb", e, d, std, seed)(ds, FrameContext.make(0))


def test_init_mean_equals_measurement_and_is_deterministic():
    z = torch.randn(1, 6)
    a = _spawn(z, 32, 6, seed=7)
    b = _spawn(z, 32, 6, seed=7)
    assert torch.allclose(a, b)  # fixed seed -> reproducible
    assert torch.allclose(a.mean(dim=1), z, atol=1e-5)  # mean-centred to z


def test_etkf_mean_equals_sample_covariance_kalman_mean():
    """ETKF analysis mean is exactly the Kalman mean using the sample covariance."""
    torch.manual_seed(2)
    d, e, r = 6, 40, 0.3
    z = torch.randn(1, 6)
    members = _spawn(torch.randn(1, 6), e, d, seed=3)
    cs = _t(members.mean(dim=1), members.clone())

    xf = cs.emb_ensemble[0]
    xbar = xf.mean(dim=0)
    anom = xf - xbar
    p_sample = anom.T @ anom / (e - 1)
    kalman_mean = xbar + p_sample @ torch.linalg.solve(
        p_sample + torch.eye(d) * r, z[0] - xbar
    )

    ds = Detections(index=torch.tensor([0]), emb=z, batch_size=[1])
    out = EnsembleUpdate("emb", "emb", r=r)(cs, ds, _match_one(), FrameContext.make(0))
    assert torch.allclose(out.emb[0], kalman_mean, atol=1e-4)
    # the analysis mean is the mean of the analysis ensemble
    assert torch.allclose(out.emb[0], out.emb_ensemble[0].mean(dim=0), atol=1e-5)


def test_update_reduces_error_toward_truth():
    torch.manual_seed(4)
    d, e = 8, 50
    truth = torch.randn(1, d)
    members = _spawn(truth + 0.4 * torch.randn(1, d), e, d, std=0.6, seed=5)
    cs = _t(members.mean(dim=1), members.clone())
    before = (cs.emb[0] - truth[0]).norm()
    z = truth + 0.05 * torch.randn(1, d)
    ds = Detections(index=torch.tensor([0]), emb=z, batch_size=[1])
    out = EnsembleUpdate("emb", "emb", r=0.05)(
        cs, ds, _match_one(), FrameContext.make(0)
    )
    after = (out.emb[0] - truth[0]).norm()
    assert after < before


def test_inflation_grows_spread_keeps_mean():
    members = _spawn(torch.zeros(1, 5), 30, 5, std=0.5, seed=1)
    cs = _t(members.mean(dim=1), members.clone())
    spread0 = cs.emb_ensemble[0].var(dim=0).sum()
    out = EnsembleProcess("emb", q=1.0)(cs, FrameContext.make(0, delta=1.0))
    spread1 = out.emb_ensemble[0].var(dim=0).sum()
    assert spread1 > spread0
    assert torch.allclose(
        out.emb_ensemble[0].mean(dim=0), cs.emb_ensemble[0].mean(dim=0), atol=1e-5
    )


def test_state_entries_keys_and_ensemble_shape():
    entries = enkf_state_entries("emb", dim=8, ensemble_size=16)
    assert set(entries) == {"emb", "emb_ensemble"}
    ds = Detections(index=torch.tensor([0, 1]), emb=torch.randn(2, 8), batch_size=[2])
    members = entries["emb_ensemble"].init(ds, FrameContext.make(0))
    assert members.shape == (2, 16, 8)
