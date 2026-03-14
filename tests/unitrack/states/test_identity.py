from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states import (
    FromDetectionField,
    Identity,
    Replace,
    ZerosInitializer,
)


def _cs(kernel: torch.Tensor) -> Tracklets:
    n = kernel.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        kernel=kernel,
        batch_size=[n],
    )


def _ds(kernel: torch.Tensor) -> Detections:
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_identity_is_a_no_op():
    cs = _cs(torch.arange(8.0).reshape(2, 4))
    out = Identity("kernel")(cs, FrameContext.make(0))
    assert torch.equal(out.kernel, cs.kernel)


def test_replace_assigns_matched_measurement_for_matched_pairs():
    cs = _cs(torch.zeros(3, 2))
    ds = _ds(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    match = MatchOutcome(
        matched_pairs=torch.tensor([[0, 1], [2, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.tensor([1], dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(2),
        batch_size=[],
    )
    out = Replace("kernel")(cs, ds, match, FrameContext.make(1))
    assert out.kernel[0].tolist() == [2.0, 2.0]  # cs row 0 <- ds row 1
    assert out.kernel[1].tolist() == [0.0, 0.0]  # unmatched, untouched
    assert out.kernel[2].tolist() == [1.0, 1.0]  # cs row 2 <- ds row 0


def test_from_detection_field_initializer():
    ds = _ds(torch.tensor([[7.0, 8.0]]))
    init = FromDetectionField("kernel")
    out = init(ds, FrameContext.make(2))
    assert out.tolist() == [[7.0, 8.0]]


def test_zeros_initializer():
    from unitrack.data import TensorSpec

    ds = _ds(torch.zeros(3, 2))
    init = ZerosInitializer(TensorSpec(shape=(4,), dtype=torch.float32))
    out = init(ds, FrameContext.make(0))
    assert out.shape == (3, 4)
    assert torch.all(out == 0)
    assert out.dtype is torch.float32
