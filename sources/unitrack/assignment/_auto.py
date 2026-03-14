"""
Smart-dispatch LAP solver routing.

Picks between the CPU LAPJV solver and the torchmatch CUDA solvers based
on the caller's explicit preference. The default (``"auto"``) routes to
the CPU path because benchmarks (see ``assets/benchmarks/``) show that
the Jonker-Volgenant CPU path is faster than the CUDA solvers across the
tested size range (N <= 256), often by 5x-25x. PCIe round-trips and
per-iteration host syncs in the matrix-form Hungarian solvers dominate
the solve cost when problems are small enough to fit in CPU L2/L3.

Use ``prefer="cuda"`` only when the cost matrix is part of a fully
GPU-resident pipeline that should not break with a host sync, or when
profiling shows the CUDA path wins for the specific workload.
"""

from __future__ import annotations

import enum
import typing

import torch

from ._base import Assignment
from .lap import lap_assignment, lap_batch_assignment
from .lapjv import lapjvx_assignment, lapjvx_batch_assignment

__all__ = ["AutoLAP", "Prefer", "auto_assignment", "auto_batch_assignment"]


class Prefer(enum.StrEnum):
    """Backend preference for :func:`auto_assignment` and :class:`AutoLAP`."""

    AUTO = "auto"
    """Pick the empirically-fastest backend. Currently always CPU LAPJV."""

    CPU = "cpu"
    """Force the CPU LAPJV solver. Same as ``AUTO`` for now."""

    CUDA = "cuda"
    """Force the torchmatch CUDA solver (Munkres/Lawler via AUTO dispatch)."""


def _require_cuda(prefer: Prefer) -> None:
    """Raise a clear error if ``prefer=CUDA`` is requested on a CUDA-less host."""
    if not torch.cuda.is_available():
        msg = (
            f"prefer={prefer.value!r} requires a CUDA-capable build of PyTorch with "
            "an available device. Use Prefer.AUTO / Prefer.CPU on this host, or "
            "install a CUDA-enabled PyTorch."
        )
        raise RuntimeError(msg)


@torch.no_grad()
def auto_assignment(
    cost: torch.Tensor,
    *,
    prefer: Prefer | str = Prefer.AUTO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve a single LAP via the empirically-fastest backend.

    Parameters
    ----------
    cost : torch.Tensor
        ``(N, M)`` cost matrix. May live on any device; non-CUDA paths
        copy to host transparently.
    prefer : Prefer or str, optional
        Backend preference. See :class:`Prefer`. Default
        :attr:`Prefer.AUTO`.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices on the
        input tensor's device.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    Raises
    ------
    RuntimeError
        If ``prefer="cuda"`` is requested without a CUDA device.

    """
    pref = Prefer(prefer)
    if pref == Prefer.CUDA:
        _require_cuda(pref)
        return lap_assignment(cost)
    return lapjvx_assignment(cost)


@torch.no_grad()
def auto_batch_assignment(
    cost_matrices: list[torch.Tensor],
    *,
    prefer: Prefer | str = Prefer.AUTO,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Solve a batch of LAPs via the empirically-fastest backend.

    Parameters
    ----------
    cost_matrices : list of torch.Tensor
        Per-problem cost matrices. Shapes may differ.
    prefer : Prefer or str, optional
        Backend preference. Default :attr:`Prefer.AUTO`.

    Returns
    -------
    list of tuple
        Per-problem ``(matches, unmatched_rows, unmatched_cols)`` triples.

    Raises
    ------
    RuntimeError
        If ``prefer="cuda"`` is requested without a CUDA device.

    """
    pref = Prefer(prefer)
    if pref == Prefer.CUDA:
        _require_cuda(pref)
        return lap_batch_assignment(cost_matrices)
    return lapjvx_batch_assignment(cost_matrices)


class AutoLAP(Assignment):
    """
    :class:`Assignment` wrapper around the auto-dispatched LAP solver.

    Defaults to the CPU LAPJV path (see module docstring). Pass
    ``prefer="cuda"`` to pin to the torchmatch CUDA solver.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`Assignment`.
    prefer : Prefer or str, optional
        Backend preference. Default :attr:`Prefer.AUTO`.
    **kwargs
        Keyword arguments forwarded to :class:`Assignment`.

    """

    prefer: typing.Final[Prefer]

    def __init__(
        self,
        *args,
        prefer: Prefer | str = Prefer.AUTO,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prefer = Prefer(prefer)

    @typing.override
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return auto_assignment(cost_matrix, prefer=self.prefer)
