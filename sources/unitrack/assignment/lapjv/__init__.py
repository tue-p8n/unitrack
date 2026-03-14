"""
CPU LAP solvers backed by :mod:`torchmatch.assignment`.

Two solver flavours:

* ``lapjvx`` -- rectangular-cost JV; the empirical default in
  :func:`unitrack.assignment.auto_assignment`.
* ``lapjvs`` -- square-cost JV; equivalent to ``lapjvx`` (torchmatch
  auto-selects the compact kernel for square inputs).
"""

from __future__ import annotations

from ._solver import (
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
    "lapjvs_assignment",
    "lapjvs_batch_assignment",
    "lapjvx_assignment",
    "lapjvx_batch_assignment",
]
