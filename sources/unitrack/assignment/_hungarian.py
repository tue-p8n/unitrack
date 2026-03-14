"""Hungarian algorithm backed by :func:`scipy.optimize.linear_sum_assignment`."""

from __future__ import annotations

import numpy as np
import torch
import torch.fx
import typing_extensions as TX  # noqa: N812
from torch import Tensor

from ._base import Assignment

__all__ = ["Hungarian", "hungarian_assignment"]


class Hungarian(Assignment):
    r"""Hungarian-algorithm LAP solver wrapping the SciPy implementation."""

    @TX.override
    def _assign(self, cost_matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Solve the assignment problem using the Hungarian algorithm.

        Parameters
        ----------
        cost_matrix : torch.Tensor
            ``(N, M)`` cost matrix.

        Returns
        -------
        tuple
            ``(matched_pairs, unmatched_rows, unmatched_cols)`` with the
            shapes documented on :meth:`Assignment.forward`.

        """
        return hungarian_assignment(cost_matrix)


def hungarian_assignment(
    cost_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve a LAP via SciPy's Hungarian implementation.

    Copies the cost matrix to host, calls
    :func:`scipy.optimize.linear_sum_assignment`, and returns the result
    as PyTorch tensors on the original device.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        ``(N, M)`` cost matrix. ``inf`` entries mark forbidden pairs;
        ``NaN`` entries raise.

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
    ValueError
        If the cost matrix contains ``NaN``.

    """
    import scipy.optimize

    device = cost_matrix.device

    cm = cost_matrix.detach().cpu().contiguous().numpy()
    if np.isnan(cm).any():
        msg = (
            "hungarian_assignment: cost matrix contains NaN. NaN is never a "
            "valid 'forbidden pair' sentinel — use +inf for that — so this "
            "indicates an upstream bug (e.g. zero-norm cosine vector, "
            "singular Kalman covariance, log of zero)."
        )
        raise ValueError(msg)

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
