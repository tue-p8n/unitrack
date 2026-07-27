r"""Bertsekas-style auction algorithm for linear assignment."""

from __future__ import annotations

import typing as T  # noqa: N812

import torch
import torch.fx
import torchmatch.assignment as _tma
import typing_extensions as TX  # noqa: N812

from ._base import Assignment

__all__ = ["Auction", "auction_assignment"]


class Auction(Assignment):
    r"""
    Bertsekas auction solver for the linear assignment problem.

    Iteratively bids unassigned rows on their most-profitable columns until
    every row holds an assignment or no further bids can be placed. Yields
    an :math:`\epsilon`-optimal matching where ``epsilon`` scales with
    ``bid_size``; smaller values approach the LAP optimum at the cost of
    more iterations.
    """

    bid_size: T.Final[float]

    def __init__(self, bid_size=0.05, *args, **kwargs):
        """
        Initialize the auction solver.

        Parameters
        ----------
        bid_size : float, optional
            Auction bid step size. Tune to the dynamic range of the cost
            matrix; values that are large relative to typical cost gaps
            converge faster but produce coarser matches.
        *args
            Positional arguments forwarded to :class:`Assignment`.
        **kwargs
            Keyword arguments forwarded to :class:`Assignment`.

        """
        super().__init__(*args, **kwargs)

        self.bid_size = bid_size

    @TX.override
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return auction_assignment(cost_matrix, self.bid_size)


@torch.no_grad()
def auction_assignment(
    cost_matrix: torch.Tensor, bid_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve a linear assignment problem using Bertsekas' auction algorithm.

    Converts the cost matrix to a profit matrix, then runs synchronous
    bidding until every row (or every column, for rectangular problems) is
    assigned. Non-finite entries mark forbidden pairs and are penalised
    internally so the solver never selects them.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        ``(N, M)`` cost matrix. ``inf`` entries mark forbidden pairs.
    bid_size : float
        Auction bid step size. The internal ``epsilon`` is derived as
        ``min(bid_size / min(N, M), 1e-3)``.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    """
    return _tma.auction_assignment(cost_matrix, bid_size)


torch.fx.wrap("auction_assignment")
