"""Greedy nearest-neighbour assignment over a cost matrix."""

from __future__ import annotations

import torch
import torch.fx
import torchmatch.assignment as _tma

from ._base import Assignment

__all__ = ["Greedy", "greedy_assignment"]


class Greedy(Assignment):
    """
    Greedy nearest-neighbour linear-assignment solver.

    Sorts cost entries ascending and consumes them in order, claiming each
    row-column pair whose endpoints are still free. The result is locally
    optimal but not globally optimal; pairs whose cost exceeds the inherited
    :attr:`Assignment.threshold` are pre-masked to ``inf`` by the base class
    and never assigned. Useful as a low-cost baseline and when the cost
    matrix is sparse enough that the optimal solution coincides with the
    greedy one (e.g. high-confidence Re-ID after motion gating).

    See :func:`.greedy_assignment` for the underlying tensor shapes and
    dtypes.
    """

    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return greedy_assignment(cost_matrix)


@torch.no_grad()
def greedy_assignment(
    cost_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign rows to columns by repeatedly picking the cheapest free pair.

    Sorts the flattened cost matrix once and walks it in ascending order,
    claiming each ``(row, col)`` whose endpoints are still free. Non-finite
    entries terminate the scan, so callers can mark forbidden pairs by
    setting them to ``inf``.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        ``(N, M)`` 2-D cost matrix.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    Notes
    -----
    Time complexity is ``O(N * M * log(N * M))``, dominated by the initial
    sort. The result is not globally optimal; for the true LAP optimum use
    :func:`.hungarian_assignment` or :func:`.auto_assignment`.

    """
    result = _tma.solve(cost_matrix, backend="greedy", unpack=True)
    return result[0], result[1], result[2]


torch.fx.wrap("greedy_assignment")
