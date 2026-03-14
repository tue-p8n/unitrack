# tests/unitrack/gates/test_soft_motion.py
from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.gates.soft import SoftMotionGate


def test_soft_motion_gate_returns_per_pair_costbias_smooth():
    cs = Tracklets(
        id=torch.zeros(1, dtype=torch.int64),
        status=torch.ones(1, dtype=torch.int8),
        hits=torch.ones(1, dtype=torch.int32),
        time_since_update=torch.zeros(1, dtype=torch.int32),
        age=torch.ones(1, dtype=torch.int32),
        frame_started=torch.zeros(1, dtype=torch.int32),
        frame_last_seen=torch.zeros(1, dtype=torch.int32),
        position=torch.tensor([[0.0, 0.0]]),
        position_cov=torch.eye(2).unsqueeze(0).contiguous(),
        batch_size=[1],
    )
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        position=torch.tensor([[0.5, 0.0], [3.0, 0.0]]),
        batch_size=[2],
    )
    g = SoftMotionGate("position", "position_cov", temperature=1.0)(
        cs, ds, FrameContext.make(0)
    )
    # Returns Gate.CostBias; closer detection has smaller bias.
    assert g.kind == "cost_bias"
    assert g.matrix[0, 0].item() < g.matrix[0, 1].item()
