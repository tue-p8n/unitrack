from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states.kalman import (
    InformationProcess,
    InformationUpdate,
    KalmanLinear,
    KalmanUpdate,
    information_state_entries,
)


def _t(fields: dict) -> Tracklets:
    n = next(iter(fields.values())).shape[0]
    base = {
        "id": torch.arange(n, dtype=torch.int64),
        "status": torch.ones(n, dtype=torch.int8),
        "hits": torch.ones(n, dtype=torch.int32),
        "time_since_update": torch.zeros(n, dtype=torch.int32),
        "age": torch.ones(n, dtype=torch.int32),
        "frame_started": torch.zeros(n, dtype=torch.int32),
        "frame_last_seen": torch.zeros(n, dtype=torch.int32),
        "batch_size": [n],
    }
    base.update(fields)
    return Tracklets(**base)


def _match_one() -> MatchOutcome:
    return MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )


def test_information_filter_matches_kalman_posterior():
    """The information filter is the exact Kalman dual; posteriors must agree."""
    torch.manual_seed(1)
    d = 6
    q, r = 0.05, 0.3
    mu0 = torch.randn(1, d)
    p0 = torch.eye(d) * 0.7 + 0.1  # SPD
    z = torch.randn(1, d)
    ctx = FrameContext.make(1, delta=1.0)
    ds = Detections(index=torch.tensor([0]), emb=z, batch_size=[1])

    # Information form: predict then update.
    y0 = torch.linalg.inv(p0).unsqueeze(0)
    yv0 = (y0 @ mu0.unsqueeze(-1)).squeeze(-1)
    inf = _t(
        {"emb": mu0.clone(), "emb_infomat": y0.clone(), "emb_infovec": yv0.clone()}
    )
    inf = InformationProcess("emb", q=q)(inf, ctx)
    inf = InformationUpdate("emb", "emb", r=r)(inf, ds, _match_one(), ctx)
    inf_mu = inf.emb[0]
    inf_cov = torch.linalg.inv(inf.emb_infomat[0])

    # Reference Kalman (F = H = I).
    kal = _t({"emb": mu0.clone(), "emb_cov": p0.clone().unsqueeze(0)})
    kal = KalmanLinear(
        field="emb",
        F=torch.eye(d),
        H=torch.eye(d),
        Q=torch.eye(d) * q,
        R=torch.eye(d) * r,
    )(kal, ctx)
    kal = KalmanUpdate(
        field="emb", cov_field="emb_cov", H=torch.eye(d), R=torch.eye(d) * r
    )(kal, ds, _match_one(), ctx)

    assert torch.allclose(inf_mu, kal.emb[0], atol=1e-4)
    assert torch.allclose(inf_cov, kal.emb_cov[0], atol=1e-4)


def test_update_is_additive_in_information_space():
    d = 4
    r = 0.5
    y0 = torch.eye(d).unsqueeze(0)
    inf = _t(
        {
            "emb": torch.zeros(1, d),
            "emb_infomat": y0.clone(),
            "emb_infovec": torch.zeros(1, d),
        }
    )
    z = torch.ones(1, d)
    ds = Detections(index=torch.tensor([0]), emb=z, batch_size=[1])
    out = InformationUpdate("emb", "emb", r=r)(
        inf, ds, _match_one(), FrameContext.make(0)
    )
    # Y' = Y + (1/r) I ; y' = y + (1/r) z
    assert torch.allclose(out.emb_infomat[0], torch.eye(d) * (1.0 + 1.0 / r), atol=1e-5)
    assert torch.allclose(out.emb_infovec[0], z[0] / r, atol=1e-5)


def test_state_entries_keys_and_init():
    entries = information_state_entries("emb", dim=4, init_var=2.0)
    assert set(entries) == {"emb", "emb_infomat", "emb_infovec"}
    ds = Detections(
        index=torch.tensor([0]),
        emb=torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        batch_size=[1],
    )
    y0 = entries["emb_infomat"].init(ds, FrameContext.make(0))
    yv0 = entries["emb_infovec"].init(ds, FrameContext.make(0))
    assert torch.allclose(y0[0], torch.eye(4) * 0.5)
    assert torch.allclose(yv0[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))  # z / init_var
