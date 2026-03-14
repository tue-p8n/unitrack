"""Associate — bridges a CostExpression to a hard assignment."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.assignment._base import Assignment
from unitrack.assignment._soft import SoftAssignment
from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    MatchOutcome,
    Tracklets,
)

__all__ = ["Associate"]


@dataclasses.dataclass(frozen=True, slots=True)
class Associate:
    """
    Materialise a :class:`~unitrack.data.CostExpression` and run an :class:`Assignment`.

    Acts as the bridge between cost-construction (gates, fused costs) and
    hard matching. For :class:`~unitrack.assignment.SoftAssignment` backends
    the Sinkhorn log-plan is solved once and reused both for hard extraction and for
    downstream :class:`~unitrack.states.SoftReplace` blending.

    Parameters
    ----------
    assignment : Assignment
        Backend that produces a hard match from a materialised cost
        matrix.

    """

    assignment: Assignment

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
        cost: CostExpression | None = None,
    ) -> MatchOutcome:
        """
        Materialise ``cost`` and produce a :class:`~unitrack.data.MatchOutcome`.

        Parameters
        ----------
        cs : Tracklets
            Current tracklet snapshot.
        ds : Detections
            Current-frame detections.
        ctx : FrameContext
            Frame context (unused; kept for protocol parity).
        cost : CostExpression, optional
            Cost expression to materialise. Required; the ``None``
            default exists only so this method matches the broader
            associator call signature.

        Returns
        -------
        MatchOutcome
            Matched pairs, residual indices, per-match costs, and
            (for soft assignment) the transport plan.

        Raises
        ------
        ValueError
            If ``cost`` is ``None``.

        """
        del ctx
        if cost is None:
            msg = "Associate requires a CostExpression input"
            raise ValueError(msg)

        n = cs.batch_size[0]
        m = ds.batch_size[0]
        if n == 0 or m == 0:
            return MatchOutcome(
                matched_pairs=torch.zeros(
                    (0, 2), dtype=torch.int64, device=cost.matrix.device
                ),
                tracklets_residual_index=torch.arange(
                    n, dtype=torch.int64, device=cost.matrix.device
                ),
                detections_residual_index=torch.arange(
                    m, dtype=torch.int64, device=cost.matrix.device
                ),
                per_match_cost=torch.zeros(
                    0, dtype=cost.matrix.dtype, device=cost.matrix.device
                ),
                batch_size=[],  # type: ignore[unknown-argument]
            )

        materialized = cost.materialize()

        # SoftAssignment: solve once and reuse the log-plan both for hard
        # match extraction and for SoftReplace's blending downstream. The
        # naïve sequence (forward → re-sinkhorn) would solve twice on
        # different inputs (masked vs unmasked) and silently disagree.
        soft_plan: torch.Tensor | None = None
        if isinstance(self.assignment, SoftAssignment):
            pairs, cs_unmatched, ds_unmatched, log_plan = (
                self.assignment.solve_with_plan(materialized)
            )
            soft_plan = log_plan.exp()
        else:
            pairs, cs_unmatched, ds_unmatched = self.assignment(materialized)

        # Post-filter: drop pairs strictly above the assignment threshold.
        # Pairs equal to the threshold are kept. Some LAP backends (e.g.
        # lapjvx) can return matches on cells that were internally set to
        # inf by the threshold guard; we reject those here so callers get
        # clean finite matches.
        if pairs.shape[0] > 0:
            pair_costs = materialized[pairs[:, 0], pairs[:, 1]]
            valid_mask = pair_costs <= self.assignment.threshold
            invalid_pairs = pairs[~valid_mask]
            pairs = pairs[valid_mask]
            per_cost = pair_costs[valid_mask]
            extra_cs = invalid_pairs[:, 0]
            extra_ds = invalid_pairs[:, 1]
            cs_unmatched = torch.cat([cs_unmatched, extra_cs]).sort().values
            ds_unmatched = torch.cat([ds_unmatched, extra_ds]).sort().values
        else:
            per_cost = torch.zeros(
                0, dtype=materialized.dtype, device=materialized.device
            )

        return MatchOutcome(
            matched_pairs=pairs,
            tracklets_residual_index=cs_unmatched,
            detections_residual_index=ds_unmatched,
            per_match_cost=per_cost,
            soft_plan=soft_plan,
            batch_size=[],  # type: ignore[reportCallIssue]
        )
