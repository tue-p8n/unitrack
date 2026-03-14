"""
Linear-assignment problem (LAP) solvers over a cost matrix.

For new code prefer :func:`auto_assignment` / :class:`AutoLAP`. Both
route to the empirically-fastest backend for the given input; the CPU
LAPJV path currently beats every CUDA solver across the benchmarked
size range (see ``assets/benchmarks/`` for the data).
"""

from __future__ import annotations

from ._auction import Auction, auction_assignment
from ._auto import AutoLAP, Prefer, auto_assignment, auto_batch_assignment
from ._base import Assignment
from ._greedy import Greedy, greedy_assignment
from ._hungarian import Hungarian, hungarian_assignment
from ._jonker import Jonker, jonker_volgenant_assignment
from ._soft import SoftAssignment, sinkhorn_log_plan, soft_assignment
from ._utils import gather_total_cost
from .associate import Associate
from .clip_associate import ClipAssociator
from .lapjv import (
    LAPJVS,
    LAPJVX,
    lapjvs_assignment,
    lapjvs_batch_assignment,
    lapjvx_assignment,
    lapjvx_batch_assignment,
)

__all__ = [
    "LAPJVS",
    "LAPJVX",
    "Assignment",
    "Associate",
    "Auction",
    "AutoLAP",
    "ClipAssociator",
    "Greedy",
    "Hungarian",
    "Jonker",
    "Prefer",
    "SoftAssignment",
    "auction_assignment",
    "auto_assignment",
    "auto_batch_assignment",
    "gather_total_cost",
    "greedy_assignment",
    "hungarian_assignment",
    "jonker_volgenant_assignment",
    "lapjvs_assignment",
    "lapjvs_batch_assignment",
    "lapjvx_assignment",
    "lapjvx_batch_assignment",
    "sinkhorn_log_plan",
    "soft_assignment",
]
