"""CPU LAPJV assignment backed by torchmatch.assignment."""

from __future__ import annotations

import typing

import torch
import torchmatch.assignment as tma

from .._base import Assignment

__all__ = [
    "LAPJVS",
    "LAPJVX",
    "lapjvs_assignment",
    "lapjvs_batch_assignment",
    "lapjvx_assignment",
    "lapjvx_batch_assignment",
]

_NATIVE_DTYPES: frozenset[torch.dtype] = frozenset({torch.float32, torch.float64})
_NEED_F64: frozenset[torch.dtype] = frozenset({torch.int32, torch.int64})


def _solver_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map any input dtype to a float dtype accepted by torchmatch."""
    if dtype in _NATIVE_DTYPES:
        return dtype
    if dtype in _NEED_F64:
        return torch.float64
    return torch.float32


@torch.no_grad()
def lapjvx_assignment(
    cost: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve a single LAP via the rectangular Jonker-Volgenant solver.

    Parameters
    ----------
    cost : torch.Tensor
        ``(N, M)`` cost matrix. May live on any device; moves to CPU
        transparently.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices on the
        input tensor's device.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    """
    device = cost.device
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.arange(n_rows, dtype=torch.long, device=device),
            torch.arange(n_cols, dtype=torch.long, device=device),
        )
    target = _solver_dtype(cost.dtype)
    cost_cpu = cost.detach().to(device="cpu", dtype=target).contiguous()
    result = tma.solve(cost_cpu, unpack=True)
    matches, unmatched_rows, unmatched_cols = result[0], result[1], result[2]
    if device.type == "cpu":
        return matches, unmatched_rows, unmatched_cols
    return (
        matches.to(device=device),
        unmatched_rows.to(device=device),
        unmatched_cols.to(device=device),
    )


@torch.no_grad()
def lapjvs_assignment(
    cost: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve a single square LAP via the Jonker-Volgenant solver.

    Parameters
    ----------
    cost : torch.Tensor
        ``(N, N)`` square cost matrix.

    Returns
    -------
    matches : torch.Tensor
        ``(N, 2)`` long tensor of matched ``(row, col)`` indices.
    unmatched_rows : torch.Tensor
        Empty long tensor (square problems leave no residuals).
    unmatched_cols : torch.Tensor
        Empty long tensor.

    """
    return lapjvx_assignment(cost)


@torch.no_grad()
def lapjvx_batch_assignment(
    cost_matrices: list[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Solve a batch of LAPs via the rectangular Jonker-Volgenant solver.

    Parameters
    ----------
    cost_matrices : list of torch.Tensor
        Per-problem cost matrices. Shapes may differ.

    Returns
    -------
    list of tuple
        Per-problem ``(matches, unmatched_rows, unmatched_cols)`` triples
        following the shape conventions of :func:`lapjvx_assignment`.

    """
    return [lapjvx_assignment(c) for c in cost_matrices]


@torch.no_grad()
def lapjvs_batch_assignment(
    cost_matrices: list[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Solve a batch of square LAPs via the Jonker-Volgenant solver.

    Parameters
    ----------
    cost_matrices : list of torch.Tensor
        Per-problem square cost matrices. Shapes may differ.

    Returns
    -------
    list of tuple
        Per-problem ``(matches, unmatched_rows, unmatched_cols)`` triples
        following the shape conventions of :func:`lapjvs_assignment`.

    """
    return [lapjvx_assignment(c) for c in cost_matrices]


class LAPJVX(Assignment):
    """
    :class:`Assignment` wrapper for the rectangular Jonker-Volgenant solver.

    See :func:`lapjvx_assignment` for shape and dtype contract.
    """

    @typing.override
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return lapjvx_assignment(cost_matrix)


class LAPJVS(Assignment):
    """
    :class:`Assignment` wrapper for the square Jonker-Volgenant solver.

    See :func:`lapjvs_assignment` for shape and dtype contract.
    """

    @typing.override
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return lapjvs_assignment(cost_matrix)
