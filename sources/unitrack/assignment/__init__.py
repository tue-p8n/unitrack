"""This package implements modules that solve a Linear Assignment Problem (LAP),
where the minimum cost must be computed over a cost-matrix.
"""

from __future__ import annotations

from ._auction import *
from ._base import *
from ._greedy import *
from ._hungarian import *
from ._jonker import *
from ._utils import *
