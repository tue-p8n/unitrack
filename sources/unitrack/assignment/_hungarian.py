"""
PyTorch implementation of the Hungarian algorithm.

Solving the assignment problem.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import torch
import torch.fx
import typing_extensions as TX  # noqa: N812
from torch import Tensor

from ._base import Assignment

__all__ = ["Hungarian", "hungarian_assignment"]


class Hungarian(Assignment):
    r"""Implements the Hungarian algorithm for solving a linear assignment problem."""

    @TX.override
    def _assign(self, cost_matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Solves the assignment problem using the Hungarian algorithm.

        Parameters
        ----------
        cost_matrix
            Cost matrix

        Returns
        -------
            Tuple of the optimal assignment and the total assignment cost.

        """
        return hungarian_assignment(cost_matrix)


def hungarian_assignment(
    cost_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform linear assingment using the SciPy implementation."""
    device = cost_matrix.device

    cm = cost_matrix.cpu().detach().contiguous()
    cm = np.where(np.isfinite(cm), cm, np.inf)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cm)
    row_ind = torch.from_numpy(row_ind).to(device=device, dtype=torch.long)
    col_ind = torch.from_numpy(col_ind).to(device=device, dtype=torch.long)

    matches = torch.column_stack((row_ind, col_ind)).long()

    idx_row = torch.arange(cm.shape[0], device=device)
    idx_col = torch.arange(cm.shape[1], device=device)

    unmatch_row = idx_row[~torch.isin(idx_row, row_ind)]
    unmatch_col = idx_col[~torch.isin(idx_col, col_ind)]

    return matches, unmatch_row, unmatch_col


torch.fx.wrap("hungarian_assignment")
