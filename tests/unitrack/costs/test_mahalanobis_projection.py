# tests/unitrack/costs/test_mahalanobis_projection.py
"""Regression tests for I2: Mahalanobis + SoftMotionGate H-projection."""

from __future__ import annotations

import torch
from unitrack.costs import Mahalanobis
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.gates import MotionGate
from unitrack.gates.soft import SoftMotionGate


def _make(cs_mean, cs_cov, ds_mean, *, mean_field="p", cov_field="p_cov"):
    n, _ = cs_mean.shape
    m, _ = ds_mean.shape
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        **{mean_field: cs_mean, cov_field: cs_cov},
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64),
        **{mean_field: ds_mean},
        batch_size=[m],
    )
    return cs, ds, FrameContext.make(0)


def test_mahalanobis_projects_state_to_meas_dim_when_state_larger():
    """6-D state vs 3-D measurement should equal the projected Mahalanobis distance."""
    cs_mean = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])  # 6-D state
    cs_cov = torch.eye(6).unsqueeze(0)  # (1, 6, 6)
    ds_mean = torch.tensor([[1.0, 2.0, 3.0]])  # 3-D measurement, identity to state head
    cs, ds, ctx = _make(cs_mean, cs_cov, ds_mean)
    cost = Mahalanobis("p", "p_cov")(cs, ds, ctx).matrix
    # diff = 0 → Mahalanobis = 0
    assert cost.item() == 0.0


def test_mahalanobis_projection_preserves_distance_on_matched_dims():
    """Offset-only state head yields distance offset^T Σ^-1 offset."""
    cs_mean = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    cs_cov = torch.eye(6).unsqueeze(0)
    ds_mean = torch.tensor([[1.0, 4.0, 3.0]])  # +2 in y only
    cs, ds, ctx = _make(cs_mean, cs_cov, ds_mean)
    cost = Mahalanobis("p", "p_cov")(cs, ds, ctx).matrix
    # diff = (0, -2, 0), Σ = I → squared distance = 4
    assert abs(cost.item() - 4.0) < 1e-5


def test_motion_gate_and_mahalanobis_share_projection_convention():
    """Same state/measurement pair: gate keeps iff cost <= max_chi2."""
    cs_mean = torch.tensor([[1.0, 2.0, 0.0, 0.0]])  # 4-D state (2D CV)
    cs_cov = torch.eye(4).unsqueeze(0)
    ds_mean = torch.tensor([[1.0, 4.0]])  # 2-D measurement
    cs, ds, ctx = _make(cs_mean, cs_cov, ds_mean)
    gate = MotionGate("p", "p_cov", max_chi2=5.0)(cs, ds, ctx)
    cost = Mahalanobis("p", "p_cov")(cs, ds, ctx).matrix
    # cost = 4 < 5 → gate mask should be True
    assert cost.item() < 5.0
    assert bool(gate.mask[0, 0]) is True


def test_soft_motion_gate_projects_to_measurement():
    """SoftMotionGate must accept 6-state vs 3-meas without dim mismatch."""
    cs_mean = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
    cs_cov = torch.eye(6).unsqueeze(0)
    ds_mean = torch.tensor([[1.0, 2.0, 3.0]])
    cs, ds, ctx = _make(cs_mean, cs_cov, ds_mean)
    gate = SoftMotionGate("p", "p_cov", temperature=1.0)(cs, ds, ctx)
    # cost_bias = 0 for matched dims with zero diff
    assert gate.matrix.item() == 0.0
