# tests/unitrack/states/kalman/test_joseph_form.py
"""Regression test for I8: KalmanUpdate uses Joseph form, preserves PSD covariance."""

from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states.kalman import KalmanUpdate


def _state(mean, cov, *, field="x", cov_field="x_cov"):
    n = mean.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        **{field: mean, cov_field: cov},
        batch_size=[n],
    )


def test_kalman_update_preserves_psd_covariance_under_repeated_update():
    """Joseph form keeps Σ symmetric PSD under accumulated rounding."""
    torch.manual_seed(0)
    d = 4
    n = 1
    m = 2
    h = torch.zeros(m, d)
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    r = torch.eye(m) * 1e-3  # near-zero observation noise — adversarial for std form

    mean = torch.zeros(n, d)
    cov = torch.eye(d).unsqueeze(0) * 0.5

    cs = _state(mean, cov, field="p", cov_field="p_cov")
    update = KalmanUpdate(field="p", cov_field="p_cov", H=h, R=r)

    for k in range(20):
        meas = torch.randn(1, m) * 0.01
        ds = Detections(
            index=torch.zeros(1, dtype=torch.int64),
            p=meas,
            batch_size=[1],
        )
        match = MatchOutcome(
            matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
            tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
            detections_residual_index=torch.zeros(0, dtype=torch.int64),
            per_match_cost=torch.zeros(1),
            batch_size=[],
        )
        cs = update(cs, ds, match, FrameContext.make(k))

        cov_now = cs.p_cov[0]
        # PSD ⇔ eigenvalues all >= 0 (within tol). Joseph form preserves this.
        eigvals = torch.linalg.eigvalsh(cov_now)
        assert torch.all(eigvals >= -1e-6), (
            f"step {k}: negative eigvals {eigvals} — Joseph form failed"
        )
        # Symmetric within tol.
        assert torch.allclose(cov_now, cov_now.transpose(-1, -2), atol=1e-6)
