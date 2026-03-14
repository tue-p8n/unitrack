# tests/unitrack/gates/test_motion.py
from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.gates import MotionGate


def test_motion_gate_isotropic_unit_cov_thresholds_squared_l2():
    n, m, d = 1, 2, 2
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        position=torch.tensor([[0.0, 0.0]]),
        position_cov=torch.eye(d).expand(n, d, d).contiguous(),
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64),
        position=torch.tensor([[1.5, 0.0], [3.0, 0.0]]),
        batch_size=[m],
    )
    g = MotionGate("position", "position_cov", max_chi2=4.0)(
        cs, ds, FrameContext.make(0)
    )
    # squared distances: 2.25, 9. Threshold=4 → keep first only.
    assert g.mask.tolist() == [[True, False]]
