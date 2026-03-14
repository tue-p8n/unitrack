r"""Helpers for downstream consumers of an assignment result."""

from __future__ import annotations

from torch import Tensor

__all__ = ["gather_total_cost"]


def gather_total_cost(cost_matrix: Tensor, assignment: Tensor) -> Tensor:
    """
    Sum the cost-matrix entries selected by a row-column assignment.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        ``(N, M)`` cost matrix.
    assignment : torch.Tensor
        ``(K, 2)`` long tensor of ``(row, col)`` index pairs.

    Returns
    -------
    torch.Tensor
        Scalar tensor holding the sum of ``cost_matrix[r, c]`` over the
        assigned pairs.

    """
    return cost_matrix[assignment[:, 0], assignment[:, 1]].sum()
