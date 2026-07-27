from __future__ import annotations

from abc import abstractmethod

import torch

__all__ = ["Assignment"]


class Assignment(torch.nn.Module):
    """
    Base class for linear-assignment-problem solvers.

    Subclasses implement ``_assign`` over a 2-D cost matrix. The base
    :meth:`forward` masks entries above :attr:`threshold` to ``inf``
    before dispatching, so callers can bound match cost without each
    backend re-implementing the guard.

    Parameters
    ----------
    threshold : float, optional
        Cost upper bound. Entries strictly above ``threshold`` are
        replaced with ``inf`` and treated as forbidden by the solver.
        Default ``inf`` (no thresholding).

    """

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
        cost_matrix : torch.Tensor
            ``(N, M)`` cost matrix to solve.

        Returns
        -------
        matched_pairs : torch.Tensor
            ``(K, 2)`` long tensor. Column 0 holds tracklet (row) indices,
            column 1 holds detection (column) indices.
        unmatched_rows : torch.Tensor
            ``(N - K,)`` tracklet indices.
        unmatched_cols : torch.Tensor
            ``(M - K,)`` detection indices.

        """
        if min(cost_matrix.shape) == 0:
            return self._no_match(cost_matrix)

        # Threshold mask coerces NaN→+inf (NaN ≤ anything is False). The
        # NaN-as-upstream-bug guard lives in each backend (hungarian on the
        # already-host numpy array, lap_assignment on the CUDA tensor, and
        # lapjv's compute_sentinel fused into the existing scan) so we avoid
        # a redundant host-sync on the hot Associate path here.
        cost_matrix = torch.where(cost_matrix <= self.threshold, cost_matrix, torch.inf)

        row_finite = torch.isfinite(cost_matrix).any(dim=1)
        if not row_finite.any():
            return self._no_match(cost_matrix)

        # Strip fully-blocked rows before solving so backends don't need to
        # handle them. Add them back as unmatched afterward.
        if not row_finite.all():
            active = torch.where(row_finite)[0]
            blocked = torch.where(~row_finite)[0]
            sub_m, sub_ur, uc = self._assign(cost_matrix[active])
            if sub_m.shape[0] > 0:
                matched = torch.stack([active[sub_m[:, 0]], sub_m[:, 1]], dim=1)
            else:
                matched = torch.empty(
                    (0, 2), dtype=torch.long, device=cost_matrix.device
                )
            ur = torch.cat([active[sub_ur], blocked]).sort().values
            return matched, ur, uc

        return self._assign(cost_matrix)

    @staticmethod
    def _no_match(
        cost_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cs_num, ds_num = cost_matrix.shape
        device = cost_matrix.device
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.arange(cs_num, dtype=torch.long, device=device),
            torch.arange(ds_num, dtype=torch.long, device=device),
        )

    @abstractmethod
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
