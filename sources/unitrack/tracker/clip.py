"""Clip-level tracker: per-frame iteration plus optional refiner mode."""

from __future__ import annotations

import dataclasses
import typing

import torch

from unitrack.data import (
    ClipDetections,
    ClipFrameContext,
    ClipMatchOutcome,
    ClipTracklets,
    Tracklets,
)
from unitrack.lifecycle import TrackletStatus

from .tracker import Tracker

__all__ = ["ClipTracker"]


@dataclasses.dataclass(slots=True)
class ClipTracker:
    """
    Wrap a :class:`~unitrack.tracker.Tracker` for clip-level inference.

    In ``mode='per_frame'``, :meth:`~unitrack.tracker.Tracker.step` iterates over the
    ``K`` clip frames. In ``mode='refine'``, a learned refiner module
    runs over the aligned :class:`~unitrack.data.ClipTracklets` after the per-frame
    pass. The output :class:`~unitrack.data.ClipTracklets` has rows aligned by
    identity (row ``n`` at frame ``k`` is the same tracklet as row ``n``
    at frame ``k + 1``); rows are padded with synthetic Removed
    placeholders when a tracklet did not exist in a given frame's raw
    snapshot, so refiners can index across frames without an ID lookup.

    Attributes
    ----------
    tracker : Tracker
        Per-frame tracker.
    mode : {'per_frame', 'refine'}
        Inference mode.
    refiner : ~collections.abc.Callable or None
        Required when ``mode='refine'``; consumes the aligned
        :class:`~unitrack.data.ClipTracklets` and the
        :class:`~unitrack.data.ClipFrameContext`.
    reset_per_clip : bool
        When ``True``, ignore the incoming snapshot and start the clip
        from an empty state with ``next_id=1``.

    """

    tracker: Tracker
    mode: typing.Literal["per_frame", "refine"] = "per_frame"
    refiner: typing.Any | None = None
    reset_per_clip: bool = False

    def process_clip(
        self,
        snapshot: Tracklets,
        clip_dets: ClipDetections,
        clip_ctx: ClipFrameContext,
    ) -> tuple[Tracklets, ClipTracklets, ClipMatchOutcome]:
        """
        Run one clip through the tracker.

        Parameters
        ----------
        snapshot
            Initial :class:`~unitrack.data.Tracklets` snapshot. Ignored when
            :attr:`reset_per_clip` is ``True``.
        clip_dets
            Per-frame :class:`~unitrack.data.ClipDetections`.
        clip_ctx
            Per-frame :class:`~unitrack.data.ClipFrameContext`.

        Returns
        -------
        tuple
            ``(final_snapshot, clip_tracklets, clip_match_outcome)``
            where ``final_snapshot`` is the last frame's snapshot (or
            the refined last frame in ``refine`` mode),
            ``clip_tracklets`` carries identity-aligned rows across
            frames, and ``clip_match_outcome`` collects per-frame raw
            matches.

        Raises
        ------
        ValueError
            If ``mode='refine'`` but :attr:`refiner` is ``None``.

        """
        snap = self.tracker.empty_snapshot() if self.reset_per_clip else snapshot
        next_id = 1 if self.reset_per_clip else _max_id_plus_one(snapshot)
        raw_snaps: list[Tracklets] = []
        per_frame_matches = []
        for k in range(clip_dets.K):
            res = self.tracker.step(
                snap, clip_dets.frames[k], clip_ctx.frame_contexts[k], next_id
            )
            snap = res.snapshot
            next_id = res.next_id
            raw_snaps.append(snap)
            per_frame_matches.append(res.match)

        aligned = _align_snapshots(raw_snaps, self.tracker)
        clip_t = ClipTracklets(frames=aligned)
        if self.mode == "refine":
            if self.refiner is None:
                msg = "ClipTracker(mode='refine') requires a refiner module"
                raise ValueError(msg)
            clip_t = self.refiner(clip_t, clip_ctx)
            snap = clip_t.frames[-1]

        return snap, clip_t, ClipMatchOutcome(frames=per_frame_matches)


def _max_id_plus_one(snap: Tracklets) -> int:
    """Return ``max(snap.id) + 1`` for non-empty input, otherwise ``1``."""
    if snap.batch_size[0] == 0:
        return 1
    return int(snap.id.max().item()) + 1


def _align_snapshots(raw_snaps: list[Tracklets], tracker: Tracker) -> list[Tracklets]:
    """
    Pad each per-frame snapshot so row ``n`` is the same identity across frames.

    Builds the union of IDs across the clip and emits one
    :class:`~unitrack.data.Tracklets` per frame whose row order follows that union.
    Frames missing a given identity get a placeholder row with status
    :attr:`TrackletStatus.Removed`, ``hits=0``, ``age=0``, and
    zero-filled user fields. The placeholder keeps the row shape stable
    so a refiner can stack frame-by-frame indices without remapping.
    """
    if not raw_snaps:
        return raw_snaps

    device = raw_snaps[0].id.device
    all_ids = torch.cat([snap.id for snap in raw_snaps if snap.batch_size[0] > 0])
    if all_ids.numel() == 0:
        return raw_snaps  # nothing to align across; empty snapshots stay empty
    union_ids = torch.unique(all_ids, sorted=True)
    n_union = int(union_ids.shape[0])

    aligned: list[Tracklets] = []
    schema_snap = tracker.empty_snapshot(device=device)
    user_field_names = list(tracker.states.keys())

    for snap in raw_snaps:
        if snap.batch_size[0] == n_union and torch.equal(snap.id, union_ids):
            aligned.append(snap)
            continue
        aligned.append(
            _pad_to_union(snap, union_ids, user_field_names, schema_snap, device)
        )
    return aligned


def _pad_to_union(
    snap: Tracklets,
    union_ids: torch.Tensor,
    user_field_names: list[str],
    schema_snap: Tracklets,
    device: torch.types.Device,
) -> Tracklets:
    """Construct a :class:`~unitrack.data.Tracklets` whose rows follow ``union_ids``."""
    n_union = int(union_ids.shape[0])
    has_rows = snap.batch_size[0] > 0
    pos = torch.full((n_union,), -1, dtype=torch.int64, device=device)
    if has_rows:
        # Tracker IDs are unique within a snapshot by construction; surface
        # any violation here because alignment would otherwise silently drop
        # rows.
        if int(torch.unique(snap.id).shape[0]) != int(snap.id.shape[0]):
            msg = (
                "_pad_to_union: snapshot has duplicate tracklet IDs — "
                "alignment would silently drop rows."
            )
            raise RuntimeError(msg)
        # O(N log N) searchsorted in place of an O(N_union * N_snap) outer
        # equality + argmax. ``union_ids`` is sorted by construction (came
        # from ``torch.unique(..., sorted=True)``); ``snap.id`` may not be.
        sorted_snap_id, sort_idx = snap.id.sort()
        n_snap = int(sorted_snap_id.shape[0])
        search_pos = torch.searchsorted(sorted_snap_id, union_ids).clamp(max=n_snap - 1)
        match_in_sorted = sorted_snap_id[search_pos] == union_ids
        pos = torch.where(match_in_sorted, sort_idx[search_pos], pos)
    present = pos >= 0

    def _scatter_scalar(name: str, dtype: torch.dtype, fill) -> torch.Tensor:
        out = torch.full((n_union,), fill, dtype=dtype, device=device)
        if has_rows and present.any():
            out[present] = getattr(snap, name)[pos[present]]
        return out

    id_col = _scatter_scalar("id", torch.int64, 0)
    id_col = torch.where(present, id_col, union_ids)
    status_col = _scatter_scalar("status", torch.int8, int(TrackletStatus.Removed))
    hits_col = _scatter_scalar("hits", torch.int32, 0)
    tsu_col = _scatter_scalar("time_since_update", torch.int32, 0)
    age_col = _scatter_scalar("age", torch.int32, 0)
    fstart_col = _scatter_scalar("frame_started", torch.int32, 0)
    flast_col = _scatter_scalar("frame_last_seen", torch.int32, 0)

    user_kwargs: dict[str, torch.Tensor] = {}
    for name in user_field_names:
        template = getattr(schema_snap, name)  # zero-row template
        trailing = template.shape[1:]
        out = torch.zeros((n_union, *trailing), dtype=template.dtype, device=device)
        if has_rows and present.any():
            out[present] = getattr(snap, name)[pos[present]]
        user_kwargs[name] = out

    return Tracklets(
        id=id_col,
        status=status_col,
        hits=hits_col,
        time_since_update=tsu_col,
        age=age_col,
        frame_started=fstart_col,
        frame_last_seen=flast_col,
        **user_kwargs,  # type: ignore[invalid-argument-type]
        batch_size=[n_union],
    )
