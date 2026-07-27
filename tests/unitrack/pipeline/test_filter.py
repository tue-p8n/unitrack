# tests/unitrack/pipeline/test_filter.py
from __future__ import annotations

import re

import pytest
import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.lifecycle import MaxAgeFilter, TrackletStatus
from unitrack.pipeline import Filter, Pipe
from unitrack.pipeline.base import PipelineTypeError


def _cs_with_ages(kernel: torch.Tensor, ages: list[int]) -> Tracklets:
    n = kernel.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.full((n,), int(TrackletStatus.Active), dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.tensor(ages, dtype=torch.int32),
        age=torch.tensor(ages, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        kernel=kernel,
        batch_size=[n],
    )


def _ds(kernel: torch.Tensor) -> Detections:
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_filter_drops_too_old_tracklets_before_pipeline():
    cs = _cs_with_ages(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ages=[0, 10])
    ds = _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    pipeline = Filter(MaxAgeFilter(max_age=4), on="cs", then=inner)

    out = pipeline(cs, ds, FrameContext.make(0))
    # The too-old tracklet (cs index 1) was filtered out, so ds index 1
    # cannot match anything; ds 0 matches cs 0. The filtered-out cs row
    # appears in tracklets_residual_index — Filter remaps indices back to
    # the unfiltered space and reports filtered rows as residual.
    assert out.matched_pairs.tolist() == [[0, 0]]
    assert sorted(out.tracklets_residual_index.tolist()) == [1]
    assert out.detections_residual_index.tolist() == [1]


def test_filter_rejects_non_associator_then_at_construction():
    """``Filter.then`` must be an Associator; misuse raises at construction."""
    cost_producer = Cosine("kernel")
    with pytest.raises(
        PipelineTypeError, match=re.escape("Filter.then must be an Associator")
    ):
        Filter(MaxAgeFilter(max_age=4), on="cs", then=cost_producer)


def test_filter_rejects_unknown_on_axis():
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    with pytest.raises(PipelineTypeError, match=re.escape("Filter.on")):
        Filter(MaxAgeFilter(max_age=4), on="oops", then=inner)  # type: ignore[arg-type]


class _LowScorePredicate:
    """Drop detections whose `score` field is below a threshold."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, ds: Detections) -> torch.Tensor:
        return ds["score"] >= self.threshold


def _ds_with_scores(kernel: torch.Tensor, scores: list[float]) -> Detections:
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64),
        kernel=kernel,
        score=torch.tensor(scores, dtype=torch.float32),
        batch_size=[m],
    )


def test_filter_on_ds_drops_low_score_detections_before_pipeline():
    """``on='ds'`` filters detections before the inner pipeline; dropped
    detections re-appear in the outer ``detections_residual_index`` so the
    caller sees them as unmatched."""
    cs = _cs_with_ages(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ages=[0, 0])
    ds = _ds_with_scores(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        scores=[0.9, 0.1],
    )
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    pipeline = Filter(_LowScorePredicate(threshold=0.5), on="ds", then=inner)

    out = pipeline(cs, ds, FrameContext.make(0))
    # ds index 1 (score 0.1) was filtered out before matching; ds 0 matches cs 0.
    # cs 1 has no detection to match against.
    assert out.matched_pairs.tolist() == [[0, 0]]
    assert sorted(out.tracklets_residual_index.tolist()) == [1]
    assert sorted(out.detections_residual_index.tolist()) == [1]


def test_filter_on_both_drops_cs_and_ds_before_pipeline():
    """``on='both'`` takes a tuple of (cs_pred, ds_pred) and filters each
    side independently; both sets of dropped rows are remapped back into
    the outer residual index space."""
    cs = _cs_with_ages(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ages=[0, 10])
    ds = _ds_with_scores(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        scores=[0.9, 0.1],
    )
    inner = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    pipeline = Filter(
        (MaxAgeFilter(max_age=4), _LowScorePredicate(threshold=0.5)),
        on="both",
        then=inner,
    )

    out = pipeline(cs, ds, FrameContext.make(0))
    # cs 1 filtered (too old) AND ds 1 filtered (low score). cs 0 matches ds 0.
    assert out.matched_pairs.tolist() == [[0, 0]]
    assert sorted(out.tracklets_residual_index.tolist()) == [1]
    assert sorted(out.detections_residual_index.tolist()) == [1]
