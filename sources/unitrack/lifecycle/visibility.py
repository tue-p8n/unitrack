"""Visibility policies."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import MatchOutcome, Tracklets

from .status import TrackletStatus

__all__ = ["ConfirmedOnly", "IncludeAll", "IncludeTentative"]


def _matched_mask(cs_n: int, match: MatchOutcome, device: torch.device) -> torch.Tensor:
    out = torch.zeros(cs_n, dtype=torch.bool, device=device)
    if match.matched_pairs.shape[0] > 0:
        out[match.matched_pairs[:, 0]] = True
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class IncludeAll:
    """Return every live tracklet's ID."""

    def __call__(self, cs: Tracklets, match: MatchOutcome) -> torch.Tensor:
        """Return the IDs of all tracklets."""
        del match
        return cs.id.clone()


@dataclasses.dataclass(frozen=True, slots=True)
class ConfirmedOnly:
    """Return IDs for Active tracklets that were matched this frame."""

    def __call__(self, cs: Tracklets, match: MatchOutcome) -> torch.Tensor:
        """Return IDs of Active, matched tracklets."""
        matched = _matched_mask(cs.batch_size[0], match, cs.id.device)
        keep = (cs.status == int(TrackletStatus.Active)) & matched
        return cs.id[keep]


@dataclasses.dataclass(frozen=True, slots=True)
class IncludeTentative:
    """Return IDs for Active or Tentative tracklets matched this frame."""

    def __call__(self, cs: Tracklets, match: MatchOutcome) -> torch.Tensor:
        """Return IDs of Active or Tentative matched tracklets."""
        matched = _matched_mask(cs.batch_size[0], match, cs.id.device)
        keep = matched & (
            (cs.status == int(TrackletStatus.Active))
            | (cs.status == int(TrackletStatus.Tentative))
        )
        return cs.id[keep]
