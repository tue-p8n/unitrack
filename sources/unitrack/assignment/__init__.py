"""
Implements modules that solve a Linear Assignment Problem (LAP).

The minimum cost must be computed over a cost-matrix.
"""

from __future__ import annotations

from ._auction import Auction
from ._base import Assignment
from ._greedy import Greedy, greedy_assignment
from ._hungarian import Hungarian, hungarian_assignment
from ._jonker import Jonker, jonker_volgenant_assignment
from ._utils import gather_total_cost

__all__ = [
    "Assignment",
    "Auction",
    "Greedy",
    "Hungarian",
    "Jonker",
    "gather_total_cost",
    "greedy_assignment",
    "hungarian_assignment",
    "jonker_volgenant_assignment",
]

