from __future__ import annotations

import torch
from unitrack.data import FrameContext, Tracklets
from unitrack.states.kalman import KalmanLinear


def _make_state(mean, cov):
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


def test_predict_advances_mean_and_grows_cov():
    f = torch.eye(2)  # constant-velocity 1D: state = [pos, vel]
    f[0, 1] = 1.0
    h = torch.tensor([[1.0, 0.0]])
    q = torch.eye(2) * 0.01
    r = torch.tensor([[0.5]])
    proc = KalmanLinear("x", F=f, H=h, Q=q, R=r)

    snap = _make_state(
        mean=torch.tensor([[0.0, 1.0]]),
        cov=torch.eye(2).expand(1, 2, 2).contiguous(),
    )
    out = proc(snap, FrameContext.make(0, delta=1.0))
    assert torch.allclose(out.x, torch.tensor([[1.0, 1.0]]))
    # dt=1 makes the Q*dt scaling a no-op vs the pre-2.0 behavior.
    expected_cov = f @ torch.eye(2) @ f.T + q
    assert torch.allclose(out.x_cov[0], expected_cov)


def _fresh_zero_snap() -> Tracklets:
    """Build a fresh zero-prior snap (TensorDict.set mutates, so each
    proc() call needs its own snap)."""
    return _make_state(
        mean=torch.zeros(1, 2),
        cov=torch.zeros(1, 2, 2),
    )


def test_predict_scales_process_noise_by_delta():
    """Q is interpreted as per-unit-time process noise; one predict step
    over `dt` accumulates exactly Q*dt into the covariance. Two predict
    steps with delta=0.5 and delta=2.0 from the same zero prior must
    produce covariances whose Q-only contribution scales by dt."""
    f = torch.eye(2)  # F=I so the F P F^T term doesn't move.
    h = torch.eye(2)
    q = torch.eye(2) * 0.1
    r = torch.eye(2)
    proc = KalmanLinear("x", F=f, H=h, Q=q, R=r)

    out_half = proc(_fresh_zero_snap(), FrameContext.make(0, delta=0.5))
    out_double = proc(_fresh_zero_snap(), FrameContext.make(0, delta=2.0))
    # P_new = F P F' + Q*dt; with P=0 and F=I, P_new = Q*dt.
    assert torch.allclose(out_half.x_cov[0], q * 0.5)
    assert torch.allclose(out_double.x_cov[0], q * 2.0)
    # Cross-check: ratio across the two dts equals the dt ratio.
    assert torch.allclose(out_double.x_cov[0], out_half.x_cov[0] * 4.0)


def test_predict_dt_scale_q_false_disables_scaling():
    """Callers that bake the per-frame Q themselves can opt out of dt-scaling."""
    f = torch.eye(2)
    h = torch.eye(2)
    q = torch.eye(2) * 0.1
    r = torch.eye(2)
    proc = KalmanLinear("x", F=f, H=h, Q=q, R=r, dt_scale_q=False)

    # With dt_scale_q=False, both dts produce the same Q contribution.
    out_half = proc(_fresh_zero_snap(), FrameContext.make(0, delta=0.5))
    out_double = proc(_fresh_zero_snap(), FrameContext.make(0, delta=2.0))
    assert torch.allclose(out_half.x_cov[0], q)
    assert torch.allclose(out_double.x_cov[0], q)
