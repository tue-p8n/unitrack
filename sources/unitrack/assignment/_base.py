from __future__ import annotations

from abc import abstractmethod

import torch

__all__ = ["Assignment"]


class Assignment(torch.nn.Module):
    """Solves a linear assignment problem (LAP)."""

    threshold: float

    def __init__(self, threshold: float = torch.inf):
        super().__init__()

        self.threshold = threshold

    @torch.jit.script_if_tracing
    def forward(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve the cost matrix.

        Parameters
        ----------
        cost_matrix
            Cost matrix (NxM) to solve

        Returns
        -------
            Tuple of matches (N_match x M_match), unmatched columns and
            unmatched rows

        """
        if min(cost_matrix.shape) == 0:
            return self._no_match(cost_matrix)

        cost_matrix = torch.where(cost_matrix < self.threshold, cost_matrix, torch.inf)

        return self._assign(cost_matrix)

    @staticmethod
    def _no_match(
        cost_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cs_num, ds_num = cost_matrix.shape
        return (
            torch.empty((0, 2), dtype=torch.long),
            torch.arange(cs_num, dtype=torch.long),
            torch.arange(ds_num, dtype=torch.long),
        )

    @abstractmethod
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
