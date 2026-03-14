from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states import EMADecay, EMAFuse


def _t(field_name: str, value: torch.Tensor) -> Tracklets:
    n = value.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        **{field_name: value},
        batch_size=[n],
    )


def test_ema_decay_halves_in_one_half_life():
    snap = _t("score", torch.tensor([1.0, 0.5]))
    out = EMADecay("score", half_life=1.0)(snap, FrameContext.make(0, delta=1.0))
    assert torch.allclose(out.score, torch.tensor([0.5, 0.25]), atol=1e-5)


def test_ema_fuse_blend_unmatched_unchanged():
    cs = _t("kernel", torch.zeros(2, 2))
    ds = Detections(
        index=torch.arange(1, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0]]),
        batch_size=[1],
    )
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.tensor([1], dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )
    out = EMAFuse("kernel", rho=0.5)(cs, ds, match, FrameContext.make(0))
    assert torch.allclose(out.kernel[0], torch.tensor([0.5, 0.0]))
    assert torch.allclose(out.kernel[1], torch.tensor([0.0, 0.0]))
