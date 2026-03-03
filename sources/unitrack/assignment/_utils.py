r"""Various utilities for working with assignment problems."""

from __future__ import annotations

from torch import Tensor

__all__ = ["gather_total_cost"]


def gather_total_cost(cost_matrix: Tensor, assignment: Tensor) -> Tensor:
    """
    Gather the total cost of an assignment.

    The amounts to summing all the assigned items from the cost matrix.

    Parameters
    ----------
    cost_matrix: Tensor[N, M]
        The cost matrix.
    assignment: Tensor[min(N, M), 2]
        The assignment tensor of row-column pairs.

    Returns
    -------
    Tensor[*]
        The total cost of the assignment.

    """
    return cost_matrix[assignment[:, 0], assignment[:, 1]].sum()
