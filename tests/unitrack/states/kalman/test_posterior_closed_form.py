"""Closed-form posterior correctness for :class:`KalmanLinear` + :class:`KalmanUpdate`.

The existing :mod:`test_joseph_form` test only asserts PSD preservation; this
file pins the *value* of the posterior against a hand-computed scalar KF and
a multi-dimensional cross-check, so a regression in the predict/update math
would surface as a numerical mismatch rather than just a degenerate covariance.
"""

from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states.kalman import KalmanLinear
from unitrack.states.kalman.update import KalmanUpdate


def _make_state(mean: torch.Tensor, cov: torch.Tensor) -> Tracklets:
    n = mean.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        x=mean,
        x_cov=cov,
        batch_size=[n],
    )


def _make_det(meas: torch.Tensor) -> Detections:
    m = meas.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64),
        x=meas,
        batch_size=[m],
    )


def test_scalar_kf_posterior_matches_closed_form():
    """1-D scalar KF: x_prior=0, P_prior=1, z=1, R=1, H=I, F=I, Q=0.

    Hand-computed posterior:
      predict: x_pp=0, P_pp=1
      S = H P_pp H^T + R = 2
      K = P_pp H^T S^{-1} = 0.5
      x_post = x_pp + K (z - H x_pp) = 0.5
      P_post = (I - K H) P_pp = 0.5  (also Joseph: 0.25 + 0.25 = 0.5)
    """
    # Use Q=0 so the predict step is the identity on covariance; otherwise
    # the Q*dt scaling in KalmanLinear would perturb the expected posterior.
    f = torch.eye(1)
    h = torch.eye(1)
    q = torch.zeros(1, 1)
    r = torch.eye(1)
    predict = KalmanLinear("x", F=f, H=h, Q=q, R=r)
    update = KalmanUpdate(field="x", cov_field="x_cov", H=h, R=r)

    snap = _make_state(
        mean=torch.tensor([[0.0]]),
        cov=torch.eye(1).expand(1, 1, 1).contiguous(),
    )
    ds = _make_det(torch.tensor([[1.0]]))
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    ctx = FrameContext.make(0, delta=1.0)
    predicted = predict(snap, ctx)
    out = update(predicted, ds, match, ctx)
    assert torch.allclose(out.x[0], torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(out.x_cov[0], torch.tensor([[0.5]]), atol=1e-6)


def test_two_dim_cv_posterior_matches_closed_form():
    """2-D constant-velocity KF (state = [pos, vel]) — observe position only.

    Predict: x_pp = F x_prior, P_pp = F P_prior F^T + Q.
    Update : S = H P_pp H^T + R, K = P_pp H^T S^{-1},
             x_post = x_pp + K (z - H x_pp),
             P_post = (I - K H) P_pp.
    """
    f = torch.tensor([[1.0, 1.0], [0.0, 1.0]])  # CV: pos += vel
    h = torch.tensor([[1.0, 0.0]])  # observe position
    q = torch.zeros(2, 2)  # no process noise, isolate the math
    r = torch.tensor([[1.0]])
    predict = KalmanLinear("x", F=f, H=h, Q=q, R=r)
    update = KalmanUpdate(field="x", cov_field="x_cov", H=h, R=r)

    p_prior = torch.eye(2)  # 2x2 identity prior
    snap = _make_state(
        mean=torch.tensor([[0.0, 1.0]]),
        cov=p_prior.expand(1, 2, 2).contiguous(),
    )
    ds = _make_det(torch.tensor([[2.0]]))
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    ctx = FrameContext.make(0, delta=1.0)
    predicted = predict(snap, ctx)
    out = update(predicted, ds, match, ctx)

    # Hand-compute the expected posterior.
    x_pp = f @ torch.tensor([0.0, 1.0])  # [1.0, 1.0]
    p_pp = f @ p_prior @ f.T  # [[2, 1], [1, 1]]
    s = h @ p_pp @ h.T + r  # [[3]]
    k = p_pp @ h.T @ torch.linalg.inv(s)  # (2, 1)
    z = torch.tensor([2.0])
    innovation = z - h @ x_pp
    expected_mean = x_pp + (k @ innovation.unsqueeze(-1)).squeeze(-1)
    expected_cov = (torch.eye(2) - k @ h) @ p_pp

    assert torch.allclose(out.x[0], expected_mean, atol=1e-5)
    assert torch.allclose(out.x_cov[0], expected_cov, atol=1e-5)
