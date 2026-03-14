# tests/unitrack/data/test_clip_stacked.py
"""Regression tests for I5: ClipDetections.stacked / frame_ranges / frame_lengths."""

from __future__ import annotations

import torch
from unitrack.data import ClipDetections, Detections


def _ds(n: int) -> Detections:
    return Detections(
        index=torch.arange(n, dtype=torch.int64),
        kernel=torch.randn(n, 2),
        batch_size=[n],
    )


def test_frame_lengths_reports_per_frame_count():
    clip = ClipDetections(frames=[_ds(2), _ds(0), _ds(3)])
    assert clip.frame_lengths.tolist() == [2, 0, 3]


def test_frame_ranges_are_contiguous_half_open_intervals():
    clip = ClipDetections(frames=[_ds(2), _ds(0), _ds(3)])
    ranges = clip.frame_ranges.tolist()
    assert ranges == [[0, 2], [2, 2], [2, 5]]


def test_stacked_concatenates_frames_with_frame_idx_marker():
    clip = ClipDetections(frames=[_ds(2), _ds(1), _ds(3)])
    stacked = clip.stacked()
    assert stacked.batch_size[0] == 2 + 1 + 3
    # frame_idx must mark each row's source frame.
    assert stacked["frame_idx"].tolist() == [0, 0, 1, 2, 2, 2]
