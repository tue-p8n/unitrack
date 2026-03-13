"""
Greedy assignment is a simple assignment algorithm.

Greedily assigns detections to tracklets, by selecting the best match at each step.

This algorithm is not guaranteed to find the optimal solution, but it is fast
and simple to implement.
"""

from __future__ import annotations

import torch
import torch.fx

from ._base import Assignment

__all__ = ["Greedy", "greedy_assignment"]


class Greedy(Assignment):
    """See :func:`.greedy_assignment` for details."""

    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return greedy_assignment(cost_matrix)


@torch.no_grad()
def greedy_assignment_legacy(
    cost_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Legacy implementation of the greedy assignment algorithm.

    See :meth:`greedy_assignment` for the new implementation.
    This legacy version is provided for reference and testing purposes.
    """
    with cost_matrix.device:
        rows, cols = cost_matrix.shape
        matches = torch.full((min(rows, cols), 2), -1, dtype=torch.long)
        unmatched_rows = torch.arange(rows, dtype=torch.long)
        unmatched_cols = torch.arange(cols, dtype=torch.long)

        match_count = 0
        while True:
            min_val, idx = torch.min(cost_matrix.flatten(), dim=0)
            row, col = idx // cols, idx % cols

            if not torch.isfinite(min_val):
                break

            matches[match_count] = torch.tensor([row, col], dtype=torch.long)
            match_count += 1

            cost_matrix[row, :] = torch.inf
            cost_matrix[:, col] = torch.inf

        unmatched_rows = unmatched_rows[torch.isfinite(cost_matrix[:, 0])]
        unmatched_cols = unmatched_cols[torch.isfinite(cost_matrix[0, :])]

    return matches[:match_count], unmatched_rows, unmatched_cols


@torch.no_grad()
def greedy_assignment(
    cost_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a greedy assignment algorithm on a cost matrix.

    Assigns pairs of elements (rows and columns) based on the minimum cost,
    with a threshold as the stopping condition.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        A 2D tensor representing the cost matrix.

    Returns
    -------
    matches : torch.Tensor
        A tensor containing the indices of matched row-column pairs.
    unmatched_rows : torch.Tensor
        A tensor containing the indices of unmatched rows.
    unmatched_cols : torch.Tensor
        A tensor containing the indices of unmatched columns.

    """
    rows, cols = cost_matrix.shape
    device = cost_matrix.device

    # Edge case: Empty matrix
    if rows == 0 or cols == 0:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.arange(rows, dtype=torch.long, device=device),
            torch.arange(cols, dtype=torch.long, device=device),
        )

    # Flatten and sort once (O(N log N) is much faster than iterative O(N^2) mins)
    flat_costs = cost_matrix.flatten()
    sorted_indices = torch.argsort(flat_costs)

    matches = []
    row_used = torch.zeros(rows, dtype=torch.bool, device=device)
    col_used = torch.zeros(cols, dtype=torch.bool, device=device)

    # Iterate through sorted costs
    for idx in sorted_indices:
        # Stop early if the lowest remaining cost is infinite (gated)
        if not torch.isfinite(flat_costs[idx]):
            break

        r, c = divmod(idx.item(), cols)

        if not row_used[r] and not col_used[c]:
            matches.append([r, c])
            row_used[r] = True
            col_used[c] = True

            # Stop early if we've matched the maximum possible pairs
            if len(matches) == min(rows, cols):
                break

    # Compile results
    if matches:
        matches_t = torch.tensor(matches, dtype=torch.long, device=device)
    else:
        matches_t = torch.empty((0, 2), dtype=torch.long, device=device)

    unmatched_rows = torch.nonzero(~row_used).squeeze(1)
    unmatched_cols = torch.nonzero(~col_used).squeeze(1)

    return matches_t, unmatched_rows, unmatched_cols


torch.fx.wrap("greedy_assignment")
torch.fx.wrap("greedy_assignment_legacy")
