"""This package implements the cost functions that are used to compute the assignment costs between tracklets and
detections.
"""

from __future__ import annotations

import enum as E
import itertools
import typing as T
from abc import abstractmethod

import torch
import typing_extensions as TX
from tensordict import TensorDictBase
from torch import nn
from torchvision.ops import box_iou

__all__ = []

# ---------------- #
# Abstract classes #
# ---------------- #


class Cost(torch.nn.Module):
    """A cost module computes an assignment cost matrix between detections and
    tracklets.
    """

    required_fields: torch.jit.Final[list[str]]

    def __init__(self, required_fields: T.Iterable[str]):
        super().__init__()

        self.required_fields = list(set(required_fields))

    @abstractmethod
    @TX.override
    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        """Computes the assignment costs between previous tracklets and current
        detections.

        This is an abstract method that should be overwritten.

        Parameters
        ----------
        cs
            Candidate observations (N).
        ds
            Detections (M).

        Returns
        -------
            Cost matrix (N x M).

        """
        raise NotImplementedError


class FieldCost(Cost):
    field: T.Final[str]
    select: T.Final[list[int]]

    def __init__(self, field: str, select: T.Iterable[int] | None = None):
        super().__init__(required_fields=[field])

        self.field = field
        select = [] if select is None else list(select)
        self.select = select

    def get_field(
        self, cs: TensorDictBase, ds: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ts_field = cs.get(self.field)
        ds_field = ds.get(self.field)

        if len(self.select) > 0:
            assert ts_field.shape[1] >= max(self.select)
            assert ds_field.shape[1] >= max(self.select)

            ts_field = ts_field[:, self.select]
            ds_field = ds_field[:, self.select]

        return ts_field, ds_field

    @TX.override
    def forward(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return self.compute(ts_field, ds_field)

    @abstractmethod
    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        """Computes the assignment costs between previous tracklets and current
        detections.

        This is an abstract method that should be overwritten.

        Parameters
        ----------
        cs
            Candidate observations (N) values for a single field.
        ds
            Detections (M) values for a single field.

        Returns
        -------
            Cost matrix (N x M).

        """
        ...


# ------------------- #
# Category gate macro #
# ------------------- #


class GateCost(FieldCost):
    """Returns a matrix where each tracklet/detection pair that have equal
    field values are set to `True` and  that have different field values are set to
    `False`.
    """

    @TX.override
    def compute(self, cs_cats: torch.Tensor, ds_cats: torch.Tensor) -> torch.Tensor:
        gate_matrix = ds_cats[None, :] - cs_cats[:, None]

        return torch.where(gate_matrix == 0, 0.0, torch.inf)

    def wrap(self, cost: Cost) -> Reduce:
        """Wraps another cost function with a gated cost function, using a sum
        reduction to merge the two costs.

        Parameters
        ----------
        cost: Cost
            The cost function to wrap.

        Returns
        -------
        Reduction
            The wrapped cost function.

        """
        return Reduce([cost, self], Reduction.SUM)


DEFAULT_EPS: T.Final = 1e-5

# ------------- #
# Overlap costs #
# ------------- #


class MaskIoU(FieldCost):
    """Computes IoU cost matrix between two sets of bitmasks."""

    eps: torch.jit.Final[float]

    def __init__(self, *args, eps: float = DEFAULT_EPS, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    @TX.override
    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        return _naive_mask_iou(cs, ds, self.eps)


class BoxIoU(MaskIoU):
    @TX.override
    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        cs_field = _pad_degenerate_boxes(cs)
        ds_field = _pad_degenerate_boxes(ds)

        iou = _complete_box_iou(cs_field, ds_field, self.eps)

        return 1.0 - iou


def _naive_mask_iou(cs: torch.Tensor, ds: torch.Tensor, eps: float) -> torch.Tensor:
    cs_m = cs.reshape(len(cs), -1).int()
    ds_m = ds.reshape(len(ds), -1).int()

    isec = torch.mm(cs_m, ds_m.T)
    area = cs_m.sum(dim=1)[:, None] + ds_m.sum(dim=1)[None, :]

    iou = (isec + eps) / (area - isec + eps)

    return 1.0 - iou


def _pad_degenerate_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Adds 1 to the far-coordinate of each box to prevent degeneracy from mask to
    box conversion.
    """
    boxes = boxes.clone()
    boxes[:, 2] += 1
    boxes[:, 3] += 1

    return boxes


def _box_diou_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """IoU with penalized center-distance."""
    iou = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = (rbi - lti).to(dtype=dtype).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2)

    # centers of boxes
    x_p = (boxes1[:, 0] + boxes1[:, 2]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 3]) / 2
    x_g = (boxes2[:, 0] + boxes2[:, 2]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # The distance between boxes' centers squared.
    centers_distance_squared = ((x_p[:, None] - x_g[None, :]).float() ** 2) + (
        (y_p[:, None] - y_g[None, :]).float() ** 2
    )

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return (
        iou - ((centers_distance_squared) / diagonal_distance_squared.clamp(eps)),
        iou,
    )


def _complete_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    boxes1 = boxes1.to(dtype=dtype)
    boxes2 = boxes2.to(dtype=dtype)

    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    w_pred = boxes1[:, None, 2] - boxes1[:, None, 0]
    h_pred = boxes1[:, None, 3] - boxes1[:, None, 1]

    w_gt = boxes2[:, 2] - boxes2[:, 0]
    h_gt = boxes2[:, 3] - boxes2[:, 1]

    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(eps)

    return diou - alpha * v


# ------------------- #
# Combinational costs #
# ------------------- #


class Weighted(Cost):
    """A weighted cost module."""

    weight: torch.Tensor

    def __init__(self, cost: Cost, weight: float):
        super().__init__(required_fields=cost.required_fields)

        self.cost = cost
        self.register_buffer(
            "weight",
            torch.as_tensor(weight, dtype=torch.float32, requires_grad=False),
            persistent=True,
        )

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        return self.weight * self.cost(cs, ds).type_as(self.weight)


class Reduction(E.StrEnum):
    """Reduction method for :class:`.Reduce` cost module."""

    SUM = E.auto()
    MEAN = E.auto()
    MIN = E.auto()
    MAX = E.auto()
    PRODUCT = E.auto()


class Reduce(Cost):
    """A cost reduction module."""

    weights: T.Final[list[float]]
    method: T.Final[method]

    def __init__(
        self,
        costs: T.Sequence[Cost],
        method: Reduction | str,
        weights: T.Sequence[float] | None = None,
    ):
        super().__init__(
            required_fields=itertools.chain(*(c.required_fields for c in costs))
        )
        self.method = Reduction(method)
        self.costs = nn.ModuleList(costs)
        if weights is None:
            weights = [1.0] * len(costs)
        self.weights = list(weights)

    @TX.override
    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        res = torch.stack(
            [fn(cs, ds) * wt for fn, wt in zip(self.costs, self.weights, strict=True)]
        )

        match self.method:
            case Reduction.SUM:
                return res.sum(dim=0)
            case Reduction.MEAN:
                return res.mean(dim=0)
            case Reduction.MIN:
                return res.min(dim=0).values
            case Reduction.MAX:
                return res.max(dim=0).values
            case Reduction.PRODUCT:
                return res.prod(dim=0)
            case _:
                pass
        msg = f"Reduction method '{method}' not implemented!"
        raise NotImplementedError(msg)


# -------------------------- #
# Various distance functions #
# -------------------------- #


class CDist(FieldCost):
    """Computes a distance between two fields using ``torch.cdist``."""

    p: T.Final[float]

    def __init__(self, *args, p_norm: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p_norm

    @TX.override
    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        return torch.cdist(cs, ds, self.p, "donot_use_mm_for_euclid_dist")


class Cosine(FieldCost):
    r"""Computes the distance between two fields based on a similarity measure with
    range $[0,1]$ by computing $1 - \mathrm{CosineSimilarity}(x,y)$.
    """

    def __init__(self, *args, eps: float = DEFAULT_EPS, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    @TX.override
    def forward(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return cosine_distance(ts_field, ds_field, self.eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps) -> torch.Tensor:
    a_norm = _stable_norm(a, eps)
    b_norm = _stable_norm(b, eps)

    return 1.0 - a_norm @ b_norm.mT


class Softmax(FieldCost):
    r"""Computes the distance between two fields based on a similarity measure with
    range $[0,1]$ by computing $1 - \mathrm{ReciprocalSoftmax}(x,y)$.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @TX.override
    def forward(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return softmax_distance(ts_field, ds_field)


def softmax_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mul = torch.mm(a, b.T)
    a2b = mul.softmax(dim=0)
    b2a = mul.softmax(dim=1)
    return 1.0 - (a2b + b2a) / 2.0


class RadialBasis(FieldCost):
    r"""Computes the radial basis function (RBF) between two fields."""

    @TX.override
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return radial_basis_distance(x, y)


def radial_basis_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.cdist(a, b, p=2))


# --------- #
# Utilities #
# --------- #


def _stable_norm(t: torch.Tensor, eps):
    norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
    return t / norm.clamp_min(eps)
