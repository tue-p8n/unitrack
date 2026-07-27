"""CUDA LAP assignment backed by torchmatch.assignment."""

from __future__ import annotations

import enum
import typing
import warnings

import torch
import torchmatch.assignment as tma

from .._base import Assignment

__all__ = ["LAP", "Backend", "lap_assignment", "lap_batch_assignment"]


class Backend(enum.StrEnum):
    """
    LAP solver backend selection.

    Maps to :class:`torchmatch.assignment.Backend` at dispatch time.
    """

    CLASSICAL = "classical"
    """Classical augmenting-path Hungarian solver (Munkres)."""

    HYBRID = "hybrid"
    """Deprecated. Falls back to :attr:`CLASSICAL` (Munkres)."""

    TREE = "tree"
    """Parallel BFS tree-augmentation Hungarian solver (Lawler)."""


def _to_tma_backend(backend: Backend) -> tma.Backend:
    if backend == Backend.HYBRID:
        warnings.warn(
            "Backend.HYBRID is deprecated and has no torchmatch equivalent; "
            "falling back to Backend.CLASSICAL (Munkres). "
            "Use Backend.CLASSICAL or Backend.TREE instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return tma.Backend.MUNKRES
    if backend == Backend.TREE:
        return tma.Backend.LAWLER
    return tma.Backend.MUNKRES


class LAP(Assignment):
    """
    Solves the linear assignment problem via torchmatch CUDA solvers.

    A CUDA-capable device is required.

    Parameters
    ----------
    backend
        Solver backend (default :attr:`Backend.CLASSICAL`). Accepts a
        :class:`Backend` member or its string value.
    threshold
        Cost threshold passed to :class:`.Assignment`.

    """

    backend: typing.Final[Backend]

    def __new__(cls, *_, **__) -> typing.Self:
        if not torch.cuda.is_available():
            msg = (
                "The `LAP` assignment requires CUDA, but no CUDA device is "
                "available in the current PyTorch build."
            )
            raise RuntimeError(msg)
        return super().__new__(cls)

    def __init__(
        self, *args, backend: Backend | str = Backend.CLASSICAL, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backend = Backend(backend)

    @typing.override
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return lap_assignment(cost_matrix, backend=self.backend)


@torch.no_grad()
def lap_assignment(
    cost_matrix: torch.Tensor,
    *,
    backend: Backend | str = Backend.CLASSICAL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve the linear assignment problem via a torchmatch CUDA solver.

    Parameters
    ----------
    cost_matrix
        ``(N, M)`` cost matrix. Non-finite entries (``inf``) mark
        forbidden assignments.
    backend
        Solver backend (default :attr:`Backend.CLASSICAL`). Accepts a
        :class:`Backend` member or its string value.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    Raises
    ------
    RuntimeError
        If no CUDA device is available.

    """
    backend = Backend(backend)
    tm_backend = _to_tma_backend(backend)

    device = cost_matrix.device
    rows, cols = cost_matrix.shape

    if rows == 0 or cols == 0:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.arange(rows, dtype=torch.long, device=device),
            torch.arange(cols, dtype=torch.long, device=device),
        )

    if not torch.cuda.is_available():
        msg = "lap_assignment requires a CUDA device (no CUDA available)"
        raise RuntimeError(msg)

    cost_cuda = (
        cost_matrix.detach()
        .to(
            device=cost_matrix.device if cost_matrix.is_cuda else "cuda",
            dtype=torch.float32,
        )
        .contiguous()
    )

    result = tma.solve(cost_cuda, backend=tm_backend, unpack=True)
    matches, unmatched_rows, unmatched_cols = result[0], result[1], result[2]

    return (
        matches.to(device=device),
        unmatched_rows.to(device=device),
        unmatched_cols.to(device=device),
    )


@torch.no_grad()
def lap_batch_assignment(
    cost_matrices: list[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Solve a batch of linear assignment problems via torchmatch CUDA solvers.

    Parameters
    ----------
    cost_matrices
        List of ``(N_i, M_i)`` cost matrices.

    Returns
    -------
    list of tuple
        Per-problem ``(matches, unmatched_rows, unmatched_cols)`` triples
        following the shape conventions of :func:`lap_assignment`.

    """
    if not cost_matrices:
        return []
    return [lap_assignment(c) for c in cost_matrices]
