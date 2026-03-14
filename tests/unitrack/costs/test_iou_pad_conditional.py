# tests/unitrack/costs/test_iou_pad_conditional.py
"""Regression test for I9: _pad_degenerate only pads zero-area boxes."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou as _tv_box_iou
from unitrack.costs import BoxIoU
from unitrack.data import Detections, FrameContext, Tracklets


def _make(cs_b, ds_b):
    n, m = cs_b.shape[0], ds_b.shape[0]
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        bbox=cs_b,
        batch_size=[n],
    )
    ds = Detections(index=torch.arange(m, dtype=torch.int64), bbox=ds_b, batch_size=[m])
    return cs, ds, FrameContext.make(0)


def test_box_iou_matches_torchvision_on_non_degenerate_boxes():
    """1 - cost should equal torchvision's box_iou on real (non-degenerate) input."""
    cs_b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    ds_b = torch.tensor([[2.0, 2.0, 8.0, 8.0], [10.0, 10.0, 20.0, 20.0]])
    cs, ds, ctx = _make(cs_b, ds_b)
    cost = BoxIoU("bbox")(cs, ds, ctx).matrix
    expected = 1.0 - _tv_box_iou(cs_b, ds_b)
    assert torch.allclose(cost, expected, atol=1e-5)


def test_box_iou_still_works_on_degenerate_zero_height_boxes():
    """Zero-height box gets +1 padding on y2; IoU well-defined."""
    cs_b = torch.tensor([[0.0, 0.0, 10.0, 0.0]])  # zero height
    ds_b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    cs, ds, ctx = _make(cs_b, ds_b)
    cost = BoxIoU("bbox")(cs, ds, ctx).matrix
    # After padding zero-height box: (0,0,10,1) vs (0,0,10,10) → IoU = 10/100 = 0.1
    assert abs(cost.item() - 0.9) < 1e-5
