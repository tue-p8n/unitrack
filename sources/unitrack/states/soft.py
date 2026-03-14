"""Differentiable :class:`Observation` companions used under ``differentiable=True``."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets

__all__ = ["SoftReplace"]


@dataclasses.dataclass(frozen=True, slots=True)
class SoftReplace:
    """
    Differentiable :class:`~unitrack.states.Replace` driven by a transport plan.

    Each tracklet's field becomes ``sum_j p[i, j] * detection.field[j]``,
    where ``p`` is an ``(N, M)`` row-stochastic transport plan. For
    ``p[i] = e_j`` (one-hot) this reduces to the hard :class:`~unitrack.states.Replace`.

    ``soft_assignment`` may be supplied at construction when the caller
    has already computed the plan; otherwise the plan is read at call
    time from ``match.soft_plan`` (attribute on :class:`~unitrack.data.MatchOutcome`),
    which :class:`~unitrack.assignment.Associate` attaches automatically
    when its assignment backend is a
    :class:`~unitrack.assignment.SoftAssignment`.

    Rows of the transport plan that sum to zero (every pair forbidden by
    an upstream gate) preserve the prior tracklet field rather than
    overwriting it with the all-zero blend that ``plan @ v`` would
    produce — that would silently destroy re-id embeddings for
    fully-gated tracklets.

    Parameters
    ----------
    field : str
        Tracklet field to blend.
    soft_assignment : torch.Tensor, optional
        Precomputed ``(N, M)`` transport plan. If ``None``, the plan is
        read from ``match.soft_plan`` at call time.

    """

    field: str
    soft_assignment: torch.Tensor | None = None

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """
        Blend each tracklet field as a weighted sum of detection values.

        Parameters
        ----------
        cs : Tracklets
            Current tracklet snapshot.
        ds : Detections
            Current-frame detections.
        match : MatchOutcome
            Matched pairs and (optionally) attached transport plan.
        ctx : FrameContext
            Frame context.

        Returns
        -------
        Tracklets
            New snapshot with the field updated.

        Raises
        ------
        RuntimeError
            If no transport plan is available — neither passed at
            construction nor attached to ``match``.

        """
        del ctx
        if cs.batch_size[0] == 0 or ds.batch_size[0] == 0:
            return cs
        plan = self.soft_assignment
        if plan is None:
            plan = _plan_from_match(match)
        if plan is None:
            msg = (
                f"SoftReplace(field={self.field!r}) has no transport plan. "
                "Pass `soft_assignment=` at construction, or pair this "
                "Observation with `Associate(SoftAssignment(...))` so the "
                "plan is attached to the MatchOutcome."
            )
            raise RuntimeError(msg)
        v = getattr(ds, self.field)  # (M, *F)
        prior = getattr(cs, self.field)
        blended = plan @ v if v.ndim == 2 else torch.einsum("nm,m...->n...", plan, v)
        # Preserve prior field where the row is fully masked (sum ≈ 0).
        row_mass = plan.sum(dim=-1)
        live = row_mass > 0
        if live.ndim < blended.ndim:
            live = live.view(live.shape + (1,) * (blended.ndim - live.ndim))
        new = torch.where(live, blended, prior)
        return cs.set(self.field, new)


def _plan_from_match(match: MatchOutcome) -> torch.Tensor | None:
    """Return the soft transport plan attached to ``match``, or ``None``."""
    return getattr(match, "soft_plan", None)
