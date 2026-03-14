"""Jonker-Volgenant LAP solver wrapping the rectangular LAPJV backend."""

import typing

import torch
import torch.fx

from ._base import Assignment
from .lapjv._solver import lapjvx_assignment

__all__ = ["Jonker", "jonker_volgenant_assignment"]


class Jonker(Assignment):
    """
    Jonker-Volgenant LAP solver backed by the rectangular LAPJV CPU backend.

    Dispatches to :func:`.lapjvx_assignment`; see that function for the
    full shape and dtype contract.
    """

    @typing.override
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return lapjvx_assignment(cost_matrix)


def jonker_volgenant_assignment(
    cost_matrix: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve a LAP via the Jonker-Volgenant algorithm with a cost threshold.

    Masks entries strictly above ``threshold`` to ``inf`` and dispatches
    to :func:`.lapjvx_assignment`.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        ``(N, M)`` cost matrix.
    threshold : float
        Cost upper bound. Entries above ``threshold`` are treated as
        forbidden pairs.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    """
    cost_matrix = torch.where(cost_matrix <= threshold, cost_matrix, torch.inf)
    return lapjvx_assignment(cost_matrix)


torch.fx.wrap("jonker_volgenant_assignment")
