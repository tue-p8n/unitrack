# tests/unitrack/data/test_clip_stacked_all_types.py
"""Stacked views on ClipTracklets, ClipFrameContext, ClipMatchOutcome."""

from __future__ import annotations

import torch
from unitrack.data import (
    ClipFrameContext,
    ClipMatchOutcome,
    ClipTracklets,
    FrameContext,
    MatchOutcome,
    StackedClipMatch,
    Tracklets,
)


def _tracklets(n: int, start: int = 0) -> Tracklets:
    return Tracklets(
        id=torch.arange(start, start + n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        batch_size=[n],
    )


def _match(k: int) -> MatchOutcome:
    return MatchOutcome(
        matched_pairs=torch.arange(2 * k, dtype=torch.int64).view(k, 2),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(k, dtype=torch.float32),
        batch_size=[],
    )


def test_clip_tracklets_frame_lengths_and_stacked():
    clip = ClipTracklets(frames=[_tracklets(2), _tracklets(0), _tracklets(3)])
    assert clip.frame_lengths.tolist() == [2, 0, 3]
    assert clip.frame_ranges.tolist() == [[0, 2], [2, 2], [2, 5]]
    stacked = clip.stacked()
    assert stacked.batch_size[0] == 5
    assert stacked["frame_idx"].tolist() == [0, 0, 2, 2, 2]


def test_clip_frame_context_frame_lengths_and_stacked():
    clip = ClipFrameContext.make(start_frame=0, K=3, fps=15.0)
    assert clip.frame_lengths.tolist() == [1, 1, 1]
    assert clip.frame_ranges.tolist() == [[0, 1], [1, 2], [2, 3]]
    stacked = clip.stacked()
    assert isinstance(stacked, FrameContext)
    assert stacked.batch_size == torch.Size([3])
    assert stacked.frame_idx.tolist() == [0, 1, 2]


def test_clip_match_outcome_frame_lengths_and_stacked():
    clip = ClipMatchOutcome(frames=[_match(2), _match(0), _match(3)])
    assert clip.frame_lengths.tolist() == [2, 0, 3]
    assert clip.frame_ranges.tolist() == [[0, 2], [2, 2], [2, 5]]
    flat = clip.stacked()
    assert isinstance(flat, StackedClipMatch)
    assert flat.matched_pairs.shape == (5, 2)
    assert flat.per_match_cost.shape == (5,)
    assert flat.frame_idx.tolist() == [0, 0, 2, 2, 2]
    assert flat.K_total == 5
