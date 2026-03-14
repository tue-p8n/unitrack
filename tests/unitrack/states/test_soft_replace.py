# tests/unitrack/states/test_soft_replace.py
from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states.soft import SoftReplace


def test_soft_replace_blends_by_per_pair_probability():
    cs = Tracklets(
        id=torch.arange(2, dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.ones(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        batch_size=[2],
    )

    # Soft assignment posterior: identity-ish (high diagonal)
    soft_assign = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    match = MatchOutcome(
        matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    out = SoftReplace("kernel", soft_assignment=soft_assign)(
        cs, ds, match, FrameContext.make(0)
    )
    # row 0: (0.9 * ds[0]) + (0.1 * ds[1]) = (1.8, 0.2)
    assert torch.allclose(out.kernel[0], torch.tensor([1.8, 0.2]), atol=1e-5)
    assert torch.allclose(out.kernel[1], torch.tensor([0.2, 1.8]), atol=1e-5)


def test_soft_replace_propagates_grad_through_transport_plan():
    """The forward blend ``new = plan @ ds.field`` must keep ``plan`` in the
    autograd graph; otherwise the soft-assignment path is silently
    non-differentiable and ``differentiable=True`` is a misleading flag.

    We back-prop through the output sum and assert that the transport plan
    receives a non-zero gradient.
    """
    cs = Tracklets(
        id=torch.arange(2, dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.ones(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        batch_size=[2],
    )
    # plan requires grad so we can detect that the blend flows gradients back.
    plan = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    match = MatchOutcome(
        matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    out = SoftReplace("kernel", soft_assignment=plan)(
        cs, ds, match, FrameContext.make(0)
    )
    assert out.kernel.requires_grad, "blended kernel must stay in the autograd graph"
    out.kernel.sum().backward()
    assert plan.grad is not None
    # Every plan entry feeds at least one output element; gradients must be
    # non-zero across the full plan.
    assert (plan.grad.abs() > 0).all()


def test_soft_replace_propagates_grad_through_detection_field():
    """Detection fields must also stay differentiable so end-to-end training
    can shape the detector via the tracking loss."""
    cs = Tracklets(
        id=torch.arange(2, dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.ones(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
    det_kernel = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=det_kernel,
        batch_size=[2],
    )
    plan = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    match = MatchOutcome(
        matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(0),
        batch_size=[],
    )
    out = SoftReplace("kernel", soft_assignment=plan)(
        cs, ds, match, FrameContext.make(0)
    )
    out.kernel.sum().backward()
    assert det_kernel.grad is not None
    assert (det_kernel.grad.abs() > 0).all()
