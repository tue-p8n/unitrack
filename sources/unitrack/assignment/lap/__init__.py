"""
CUDA LAP solvers backed by :mod:`torchmatch.assignment`.

Three backends (classical/Munkres, tree/Lawler, hybrid/deprecated) plus
a batched entry point. Prefer
:func:`unitrack.assignment.auto_assignment` /
:class:`unitrack.assignment.AutoLAP` for new code; reach for these
backends only when you need explicit CUDA solver control.
"""

from __future__ import annotations

from ._solver import LAP, Backend, lap_assignment, lap_batch_assignment

__all__ = ["LAP", "Backend", "lap_assignment", "lap_batch_assignment"]
