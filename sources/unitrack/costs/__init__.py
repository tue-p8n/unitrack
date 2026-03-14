"""
Cost producers and cost-side combinators.

Leaves consume a :class:`~unitrack.data.Tracklets` /
:class:`~unitrack.data.Detections` pair and emit a
:class:`~unitrack.data.CostExpression`. The lower-is-better convention
holds throughout: a cost of ``0`` is a perfect match.
"""

from __future__ import annotations

from .combinators import Reduce, Reduction, Sinkhorn, Weighted
from .distance import RBF, BiSoftmax, CDist, Chamfer, Cosine, Mahalanobis
from .gallery import GalleryCost
from .overlap import BoxCIoU, BoxGIoU, BoxIoU, MaskIoU

__all__ = [
    "RBF",
    "BiSoftmax",
    "BoxCIoU",
    "BoxGIoU",
    "BoxIoU",
    "CDist",
    "Chamfer",
    "Cosine",
    "GalleryCost",
    "Mahalanobis",
    "MaskIoU",
    "Reduce",
    "Reduction",
    "Sinkhorn",
    "Weighted",
]
