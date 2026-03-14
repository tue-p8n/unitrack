"""
Differentiable soft assignment via Sinkhorn iterations.

Provides :class:`.SoftAssignment`, an :class:`.Assignment` subclass that
solves an entropy-regularized optimal-transport problem on the cost
matrix and extracts discrete matches from the resulting transport plan.
The soft plan is itself differentiable with respect to the input cost
matrix; the hard-match extraction step is not.
"""

from __future__ import annotations

import typing

import torch
import torch.fx
import torchmatch.transport.matrix as _tmt

from ._base import Assignment
from ._hungarian import hungarian_assignment

__all__ = ["SoftAssignment", "sinkhorn_log_plan", "soft_assignment"]


DEFAULT_EPSILON: typing.Final[float] = 0.1
DEFAULT_NUM_ITER: typing.Final[int] = 50


class SoftAssignment(Assignment):
    """
    Differentiable linear-assignment solver using Sinkhorn iterations.

    Computes an entropy-regularized optimal-transport plan for the cost
    matrix (via log-domain Sinkhorn iterations) and extracts a discrete
    assignment by taking mutual-argmax pairs over the plan. The plan
    itself is fully differentiable; use :func:`.sinkhorn_log_plan`
    directly when training with a soft-assignment loss.
    """

    epsilon: typing.Final[float]
    num_iter: typing.Final[int]

    def __init__(
        self,
        *args,
        epsilon: float = DEFAULT_EPSILON,
        num_iter: int = DEFAULT_NUM_ITER,
        **kwargs,
    ):
        """
        Initialize the soft-assignment module.

        Parameters
        ----------
        epsilon
            Entropy-regularization weight. Smaller values yield sharper,
            more discrete transport plans at the cost of slower Sinkhorn
            convergence and reduced numerical stability.
        num_iter
            Number of Sinkhorn iterations.
        *args
            Positional arguments passed to :class:`.Assignment`.
        **kwargs
            Keyword arguments passed to :class:`.Assignment`.

        """
        if epsilon <= 0:
            msg = f"SoftAssignment: epsilon must be positive; got {epsilon!r}."
            raise ValueError(msg)
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.num_iter = num_iter

    @typing.override
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return soft_assignment(
            cost_matrix, epsilon=self.epsilon, num_iter=self.num_iter
        )

    def solve_with_plan(
        self, cost_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve and return matches together with the Sinkhorn log-plan.

        Threads the Sinkhorn log-plan back to the caller in one solve.
        Use this when a downstream consumer (e.g. :class:`~unitrack.states.SoftReplace`)
        needs the same plan that produced the matches — calling
        :meth:`~unitrack.assignment.Assignment.forward` and re-running
        :func:`sinkhorn_log_plan` would
        compute the plan twice on different inputs (the threshold-masked
        cost vs the un-masked original).

        Parameters
        ----------
        cost_matrix : torch.Tensor
            ``(N, M)`` cost matrix.

        Returns
        -------
        matches : torch.Tensor
            ``(K, 2)`` long tensor of matched ``(row, col)`` indices.
        unmatched_rows : torch.Tensor
            ``(N - K,)`` long tensor of unmatched row indices.
        unmatched_cols : torch.Tensor
            ``(M - K,)`` long tensor of unmatched column indices.
        log_plan : torch.Tensor
            ``(N, M)`` log of the Sinkhorn transport plan.

        """
        device = cost_matrix.device
        dtype = cost_matrix.dtype
        if min(cost_matrix.shape) == 0:
            matches, unmatched_rows, unmatched_cols = self._no_match(cost_matrix)
            log_plan = torch.empty(cost_matrix.shape, device=device, dtype=dtype)
            return matches, unmatched_rows, unmatched_cols, log_plan

        masked = torch.where(cost_matrix <= self.threshold, cost_matrix, torch.inf)
        log_plan = sinkhorn_log_plan(
            masked, epsilon=self.epsilon, num_iter=self.num_iter
        )
        matches, unmatched_rows, unmatched_cols = hungarian_assignment(-log_plan)
        if matches.shape[0] > 0:
            pair_costs = masked[matches[:, 0], matches[:, 1]]
            valid = torch.isfinite(pair_costs)
            if not bool(valid.all()):
                invalid_pairs = matches[~valid]
                matches = matches[valid]
                unmatched_rows = (
                    torch.cat([unmatched_rows, invalid_pairs[:, 0]]).sort().values
                )
                unmatched_cols = (
                    torch.cat([unmatched_cols, invalid_pairs[:, 1]]).sort().values
                )
        return matches, unmatched_rows, unmatched_cols, log_plan


def sinkhorn_log_plan(
    cost_matrix: torch.Tensor,
    epsilon: float = DEFAULT_EPSILON,
    num_iter: int = DEFAULT_NUM_ITER,
    row_marginal: torch.Tensor | None = None,
    col_marginal: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""
    Compute a log-domain Sinkhorn transport plan for a cost matrix.

    Performs ``num_iter`` iterations of log-domain Sinkhorn updates on
    the Gibbs kernel :math:`\log K = -C / \epsilon`, producing the log
    of the entropy-regularized optimal transport plan. The result is
    numerically stable with respect to ``inf`` entries in ``C`` (which
    mark forbidden assignments) via ``torch.logsumexp``.

    Parameters
    ----------
    cost_matrix
        ``(N, M)`` cost matrix. ``inf`` entries are treated as forbidden
        assignments and produce ``-inf`` entries in the log-plan.
    epsilon
        Entropy-regularization weight.
    num_iter
        Number of Sinkhorn iterations.
    row_marginal
        ``(N,)`` target row marginal. Uniform ``1/N`` by default.
    col_marginal
        ``(M,)`` target column marginal. Uniform ``1/M`` by default.

    Returns
    -------
    torch.Tensor
        ``(N, M)`` tensor of ``log P`` where ``P`` is the Sinkhorn plan.

    Raises
    ------
    ValueError
        If ``epsilon`` is not strictly positive.

    Notes
    -----
    The Gibbs kernel is built in the log domain, so ``inf`` costs become
    ``-inf`` entries which :func:`torch.logsumexp` correctly treats as a
    zero contribution (``exp(-inf) == 0``).

    """
    if epsilon <= 0:
        msg = (
            f"sinkhorn_log_plan: epsilon must be positive; got {epsilon!r}. "
            "Smaller epsilon sharpens the plan; use 1e-3 as a practical lower bound."
        )
        raise ValueError(msg)
    return _tmt.solve(
        cost_matrix,
        a=row_marginal,
        b=col_marginal,
        reg=epsilon,
        n_iter=num_iter,
    )


def soft_assignment(
    cost_matrix: torch.Tensor,
    epsilon: float = DEFAULT_EPSILON,
    num_iter: int = DEFAULT_NUM_ITER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Derive a discrete assignment from a Sinkhorn transport plan.

    Runs :func:`.sinkhorn_log_plan` on the cost matrix, then extracts a
    full discrete assignment by running the Hungarian algorithm on
    ``-log_plan`` — equivalently, maximising the plan's log-likelihood
    over a bipartite matching. Pairs whose underlying cost is non-finite
    are rejected and moved to the residual.

    The hard extraction is non-differentiable. When training with a
    soft-assignment loss, call :func:`.sinkhorn_log_plan` directly and
    compute the loss from the returned log-plan.

    Parameters
    ----------
    cost_matrix
        ``(N, M)`` cost matrix. ``inf`` entries mark forbidden pairs.
    epsilon
        Entropy-regularization weight, forwarded to Sinkhorn.
    num_iter
        Number of Sinkhorn iterations, forwarded to Sinkhorn.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    """
    rows, cols = cost_matrix.shape
    device = cost_matrix.device

    if rows == 0 or cols == 0:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.arange(rows, dtype=torch.long, device=device),
            torch.arange(cols, dtype=torch.long, device=device),
        )

    log_plan = sinkhorn_log_plan(cost_matrix, epsilon, num_iter)

    # Hungarian on -log_plan to recover the optimal discrete assignment.
    # `linear_sum_assignment` minimises, so negate; -inf entries in log_plan
    # (forbidden pairs) become +inf — already masked-out for the matcher.
    matches, unmatched_rows, unmatched_cols = hungarian_assignment(-log_plan)

    # Drop matches whose underlying cost is non-finite (forbidden pair that
    # the matcher still picked because everything else was also forbidden).
    if matches.shape[0] > 0:
        pair_costs = cost_matrix[matches[:, 0], matches[:, 1]]
        valid = torch.isfinite(pair_costs)
        if not bool(valid.all()):
            invalid_pairs = matches[~valid]
            matches = matches[valid]
            unmatched_rows = (
                torch.cat([unmatched_rows, invalid_pairs[:, 0]]).sort().values
            )
            unmatched_cols = (
                torch.cat([unmatched_cols, invalid_pairs[:, 1]]).sort().values
            )

    return matches, unmatched_rows, unmatched_cols


torch.fx.wrap("sinkhorn_log_plan")
torch.fx.wrap("soft_assignment")
