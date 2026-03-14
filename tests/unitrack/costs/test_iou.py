# tests/unitrack/costs/test_iou.py
from __future__ import annotations

import torch
from unitrack.costs import BoxCIoU, BoxGIoU, BoxIoU, MaskIoU
from unitrack.data import Detections, FrameContext, Tracklets


def _boxes(cs_b, ds_b):
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


def test_box_iou_identical_boxes_zero_cost():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    cs, ds, ctx = _boxes(boxes, boxes)
    cost = BoxIoU("bbox")(cs, ds, ctx).matrix
    assert cost.item() < 1e-5


def test_box_iou_disjoint_boxes_unit_cost():
    cs_b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    ds_b = torch.tensor([[10.0, 10.0, 11.0, 11.0]])
    cs, ds, ctx = _boxes(cs_b, ds_b)
    cost = BoxIoU("bbox")(cs, ds, ctx).matrix
    assert abs(cost.item() - 1.0) < 1e-5


def test_box_giou_disjoint_boxes_greater_than_one():
    cs_b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    ds_b = torch.tensor([[10.0, 10.0, 11.0, 11.0]])
    cs, ds, ctx = _boxes(cs_b, ds_b)
    cost = BoxGIoU("bbox")(cs, ds, ctx).matrix
    # GIoU ∈ [-1, 1]; cost = 1 - GIoU, so disjoint with large enclosing
    # area gives a cost > 1.
    assert cost.item() > 1.0


def test_box_ciou_identical_boxes_zero_cost():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    cs, ds, ctx = _boxes(boxes, boxes)
    cost = BoxCIoU("bbox")(cs, ds, ctx).matrix
    assert cost.item() < 1e-5


def test_mask_iou_identical_masks_zero_cost():
    n = m = 1
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, 1:3, 1:3] = True
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        mask=mask,
        batch_size=[n],
    )
    ds = Detections(index=torch.arange(m, dtype=torch.int64), mask=mask, batch_size=[m])
    cost = MaskIoU("mask")(cs, ds, FrameContext.make(0)).matrix
    assert cost.item() < 1e-5


def test_mask_iou_both_empty_masks_give_unit_cost():
    """Two all-zero masks carry no overlap information; cost must be 1.0,
    not 0.0 (the eps/eps limit would otherwise treat them as identical)."""
    n = m = 1
    empty = torch.zeros(1, 4, 4, dtype=torch.bool)
    cs = Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        mask=empty,
        batch_size=[n],
    )
    ds = Detections(
        index=torch.arange(m, dtype=torch.int64), mask=empty, batch_size=[m]
    )
    cost = MaskIoU("mask")(cs, ds, FrameContext.make(0)).matrix
    assert abs(cost.item() - 1.0) < 1e-5
