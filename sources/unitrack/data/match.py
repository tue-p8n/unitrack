"""Result of running a stage tree to assignment."""

from __future__ import annotations

import torch
from tensordict import tensorclass

__all__ = ["MatchOutcome"]


@tensorclass
class MatchOutcome:
    """
    Output of a stage subtree rooted at an Associator.

    Each row of ``matched_pairs`` records one matched tracklet-detection
    pair; the residual indices identify cs/ds rows that no stage in the
    subtree could place. The :class:`~unitrack.pipeline.Associator`
    protocol returns one of these per call.

    Attributes
    ----------
    matched_pairs : torch.Tensor
        ``int64`` shape ``(K, 2)``. Row ``k`` is ``(cs_idx, ds_idx)``
        addressing the input snapshot and detections.
    tracklets_residual_index : torch.Tensor
        ``int64`` shape ``(P,)`` indices into the input ``cs`` for
        tracklets left unmatched by this subtree.
    detections_residual_index : torch.Tensor
        ``int64`` shape ``(Q,)`` indices into the input ``ds`` for
        detections left unmatched by this subtree.
    per_match_cost : torch.Tensor
        Float shape ``(K,)`` cost realised on each matched pair.
    soft_plan : torch.Tensor or None
        Float ``(N, M)`` row-stochastic transport plan when the outcome
        came from a soft (differentiable) associator; ``None``
        otherwise. Soft observations (e.g.
        :class:`~unitrack.states.SoftReplace`) read it to blend tracklet
        fields with the same plan that drove the matching.

    """

    matched_pairs: torch.Tensor
    tracklets_residual_index: torch.Tensor
    detections_residual_index: torch.Tensor
    per_match_cost: torch.Tensor
    soft_plan: torch.Tensor | None = None

    @classmethod
    def empty(cls, *, device: torch.types.Device | None = None) -> MatchOutcome:
        """
        Construct an empty :class:`~unitrack.data.MatchOutcome`.

        Parameters
        ----------
        device
            Device for the zero-row tensors.

        Returns
        -------
        MatchOutcome
            All fields are zero-length tensors and ``soft_plan`` is
            ``None``.

        """
        return cls(
            matched_pairs=torch.zeros((0, 2), dtype=torch.int64, device=device),
            tracklets_residual_index=torch.zeros(0, dtype=torch.int64, device=device),
            detections_residual_index=torch.zeros(0, dtype=torch.int64, device=device),
            per_match_cost=torch.zeros(0, dtype=torch.float32, device=device),
            soft_plan=None,
            batch_size=[],  # type: ignore[unknown-argument]
        )
