"""IoU-family :class:`~unitrack.pipeline.CostProducer` leaves."""

from __future__ import annotations

import dataclasses

import torch
from torchvision.ops import box_iou as _tv_box_iou
from torchvision.ops import complete_box_iou as _tv_complete_box_iou
from torchvision.ops import generalized_box_iou as _tv_generalized_box_iou

from unitrack.data import CostExpression, Detections, FrameContext, Tracklets

from .distance import _get_field

__all__ = ["BoxCIoU", "BoxGIoU", "BoxIoU", "MaskIoU"]


@dataclasses.dataclass(frozen=True, slots=True)
class MaskIoU:
    """
    ``1 - IoU`` cost over a named boolean mask field.

    Attributes
    ----------
    field : str
        Name of the boolean mask field on both ``cs`` and ``ds``.
    eps : float
        Numerical guard for the ratio; both empty masks are treated as
        cost ``1`` (no information) rather than ``0``.

    """

    field: str
    eps: float = 1e-5

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the mask-IoU cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix bounded in ``[0, 1]``; lower is better.

        """
        del ctx
        # int64 accumulator: int32 matmul can overflow for large H*W*N.
        a = _get_field(cs, self.field).flatten(1).to(torch.int64)
        b = _get_field(ds, self.field).flatten(1).to(torch.int64)
        isec = a @ b.T
        area = a.sum(dim=1, keepdim=True) + b.sum(dim=1, keepdim=True).T
        # Pairs where both masks are empty have area==0; without a guard the
        # (eps/eps)=1 limit collapses cost to 0 (a perfect match), which is
        # the opposite of "no information". Treat such pairs as cost=1.0.
        iou = (isec + self.eps) / (area - isec + self.eps)
        both_empty = area == 0
        iou = torch.where(both_empty, torch.zeros_like(iou), iou)
        return CostExpression.from_matrix(1.0 - iou)


def _pad_degenerate(boxes: torch.Tensor) -> torch.Tensor:
    """
    Inflate zero-width/zero-height boxes by one pixel so IoU is well-defined.

    Adds ``1`` to ``x2`` / ``y2`` only where ``x2 <= x1`` or
    ``y2 <= y1``, so non-degenerate boxes pass through unchanged and the
    result matches :func:`torchvision.ops.box_iou` bit-for-bit on real
    inputs.
    """
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    degen_w = boxes[:, 2] <= boxes[:, 0]
    degen_h = boxes[:, 3] <= boxes[:, 1]
    boxes[:, 2] = torch.where(degen_w, boxes[:, 2] + 1, boxes[:, 2])
    boxes[:, 3] = torch.where(degen_h, boxes[:, 3] + 1, boxes[:, 3])
    return boxes


@dataclasses.dataclass(frozen=True, slots=True)
class BoxIoU:
    """
    ``1 - IoU`` cost over a named bounding-box field (``xyxy`` format).

    Attributes
    ----------
    field : str
        Name of the bounding-box field on both ``cs`` and ``ds``.

    """

    field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the box-IoU cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix bounded in ``[0, 1]``; lower is better.

        """
        del ctx
        a = _pad_degenerate(_get_field(cs, self.field).float())
        b = _pad_degenerate(_get_field(ds, self.field).float())
        iou = _tv_box_iou(a, b)
        return CostExpression.from_matrix(1.0 - iou)


@dataclasses.dataclass(frozen=True, slots=True)
class BoxGIoU:
    """
    ``1 - GIoU`` cost over a named bounding-box field (``xyxy`` format).

    Attributes
    ----------
    field : str
        Name of the bounding-box field on both ``cs`` and ``ds``.

    """

    field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the generalised box-IoU cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix; lower is better. Unlike box-IoU,
            non-overlapping pairs receive a finite positive cost rather
            than the constant ``1``.

        """
        del ctx
        a = _pad_degenerate(_get_field(cs, self.field).float())
        b = _pad_degenerate(_get_field(ds, self.field).float())
        giou = _tv_generalized_box_iou(a, b)
        return CostExpression.from_matrix(1.0 - giou)


@dataclasses.dataclass(frozen=True, slots=True)
class BoxCIoU:
    """
    ``1 - CIoU`` cost over a named bounding-box field (``xyxy`` format).

    Attributes
    ----------
    field : str
        Name of the bounding-box field on both ``cs`` and ``ds``.

    """

    field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the complete box-IoU cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix; lower is better. CIoU adds a
            centre-distance and aspect-ratio penalty on top of GIoU.

        """
        del ctx
        a = _pad_degenerate(_get_field(cs, self.field).float())
        b = _pad_degenerate(_get_field(ds, self.field).float())
        ciou = _tv_complete_box_iou(a, b)
        return CostExpression.from_matrix(1.0 - ciou)
