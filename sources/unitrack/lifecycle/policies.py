"""Lifecycle policies."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import FrameContext, MatchOutcome, Tracklets

from .status import TrackletStatus

__all__ = ["NoLifecycle", "StandardLifecycle"]


def _matched_mask(n: int, match: MatchOutcome, device: torch.device) -> torch.Tensor:
    """``(n,)`` bool mask: True where the tracklet appears in matched_pairs."""
    out = torch.zeros(n, dtype=torch.bool, device=device)
    if match.matched_pairs.shape[0] > 0:
        out[match.matched_pairs[:, 0]] = True
    return out


def _apply_transitions(  # noqa: PLR0913 — state-machine config; bundling adds boilerplate
    cs: Tracklets,
    match: MatchOutcome,
    ctx: FrameContext,
    *,
    min_hits: int,
    max_age: int,
    grace_period: int,
    allow_reid: int,
) -> tuple[Tracklets, torch.Tensor]:
    """
    Run the StandardLifecycle state machine and return ``(updated_cs, removed_mask)``.

    Shared core between :class:`~unitrack.lifecycle.StandardLifecycle`
    (which filters Removed rows) and :class:`~unitrack.lifecycle.SoftLifecycle`
    (which keeps every row for shape-stable autograd). Status transitions are
    intrinsically discrete; the soft path only avoids the row-removal step.
    """
    n = cs.batch_size[0]
    matched = _matched_mask(n, match, cs.id.device)
    # Newly-spawned tracklets (age==0 entering this policy step) were created
    # from a detection on this frame; they're "matched by construction" even
    # though they aren't in match.matched_pairs (which references pre-init
    # tracklet indices only).
    is_new = cs.age == 0
    matched = matched | is_new

    status = cs.status.clone()
    orig_status = cs.status.clone()
    hits = cs.hits.clone()
    tsu = cs.time_since_update.clone()
    age = cs.age + 1

    # New tracklets already have hits=1 from init; don't double-count.
    hits = torch.where(matched & ~is_new, hits + 1, hits)
    tsu = torch.where(matched, torch.zeros_like(tsu), tsu + 1)

    is_tent = status == int(TrackletStatus.Tentative)
    promote = is_tent & (hits >= min_hits) & matched
    status = torch.where(
        promote, torch.full_like(status, int(TrackletStatus.Active)), status
    )

    # Re-id: Lost + match within allow_reid → Active. Uses orig_status so
    # tracklets demoted to Lost earlier in this same call don't re-promote
    # on the same frame they were demoted.
    if allow_reid > 0:
        reid = (orig_status == int(TrackletStatus.Lost)) & matched & ~is_new
        status = torch.where(
            reid, torch.full_like(status, int(TrackletStatus.Active)), status
        )

    remove_tent = is_tent & ~matched & (age > grace_period)
    status = torch.where(
        remove_tent, torch.full_like(status, int(TrackletStatus.Removed)), status
    )

    to_lost = (status == int(TrackletStatus.Active)) & (tsu > max_age)
    status = torch.where(
        to_lost, torch.full_like(status, int(TrackletStatus.Lost)), status
    )

    max_tsu = max_age + allow_reid
    to_remove = (orig_status == int(TrackletStatus.Lost)) & (tsu > max_tsu)
    status = torch.where(
        to_remove, torch.full_like(status, int(TrackletStatus.Removed)), status
    )

    fls = cs.frame_last_seen.clone()
    fls = torch.where(matched, ctx.frame_idx.to(fls.dtype), fls)

    updated = (
        cs.set("status", status)
        .set("hits", hits)
        .set("time_since_update", tsu)
        .set("age", age)
        .set("frame_last_seen", fls)
    )
    removed = status == int(TrackletStatus.Removed)
    return updated, removed


def _stamp_last_seen(
    cs: Tracklets, match: MatchOutcome, ctx: FrameContext
) -> Tracklets:
    """Stamp ``frame_last_seen`` on matched rows. No-op when nothing matches."""
    if cs.batch_size[0] == 0 or match.matched_pairs.shape[0] == 0:
        return cs
    fls = cs.frame_last_seen.clone()
    fls[match.matched_pairs[:, 0]] = ctx.frame_idx.to(fls.dtype)
    return cs.set("frame_last_seen", fls)


@dataclasses.dataclass(frozen=True, slots=True)
class NoLifecycle:
    """No-op policy. Tracklets stay in whatever status they entered with."""

    def __call__(
        self,
        cs: Tracklets,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Stamp frame_last_seen on matched rows; otherwise unchanged."""
        return _stamp_last_seen(cs, match, ctx)


@dataclasses.dataclass(frozen=True, slots=True)
class StandardLifecycle:
    """
    Standard tracklet lifecycle policy.

    Transitions per frame (in order):

    - Tentative + match → Active when ``hits >= min_hits``.
    - Tentative + miss → Removed when ``age >= grace_period`` (default 0:
      immediate removal). With ``grace_period > 0`` a fresh Tentative that
      misses on its first few frames stays Tentative until the grace window
      expires, which gives the matcher a chance to recover a flickering
      detection.
    - Lost + match within ``allow_reid`` → Active (re-id promotion).
    - Active + ``tsu > max_age`` → Lost.
    - Lost beyond ``max_age + allow_reid`` → Removed.

    All transitions take effect before the Removed rows are filtered out.
    """

    min_hits: int
    max_age: int
    grace_period: int = 0
    allow_reid: int = 0

    def __call__(
        self,
        cs: Tracklets,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Apply lifecycle transitions and filter out Removed tracklets."""
        if cs.batch_size[0] == 0:
            return cs
        updated, removed = _apply_transitions(
            cs,
            match,
            ctx,
            min_hits=self.min_hits,
            max_age=self.max_age,
            grace_period=self.grace_period,
            allow_reid=self.allow_reid,
        )
        return updated[~removed]
