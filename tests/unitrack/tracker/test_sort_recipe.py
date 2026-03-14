"""Smoke test for the SORT recipe in docs/recipes/sort.md.

Pins two things:
- The recipe runs end-to-end across multiple frames (the bbox_xyxy
  projection state from the recipe must keep both the IoU cost happy
  and the IDs stable under matching).
- ``ConfirmedOnly`` returns the freshly-spawned tracklets on their
  first frame when ``min_hits=1`` (regression for the first-frame
  visibility hole).
"""

from __future__ import annotations

import dataclasses

import torch
import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import BoxIoU
from unitrack.data import Detections, FrameContext, MatchOutcome, TensorSpec, Tracklets
from unitrack.lifecycle import ConfirmedOnly, StandardLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, State
from unitrack.states.kalman import KalmanBBox


def _xyah_to_xyxy(xyah: torch.Tensor) -> torch.Tensor:
    cx, cy, a, h = xyah[..., 0], xyah[..., 1], xyah[..., 2], xyah[..., 3]
    w = a * h
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


@dataclasses.dataclass(frozen=True, slots=True)
class _ProjectXyahToXyxy:
    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        del ctx
        if cs.batch_size[0] == 0:
            return cs
        return cs.set("bbox_xyxy", _xyah_to_xyxy(cs.bbox_xyah))


@dataclasses.dataclass(frozen=True, slots=True)
class _ProjectXyahToXyxyAfterUpdate:
    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        del ds, match, ctx
        if cs.batch_size[0] == 0:
            return cs
        return cs.set("bbox_xyxy", _xyah_to_xyxy(cs.bbox_xyah))


def _build_sort_tracker(min_hits: int) -> unitrack.Tracker:
    bbox = KalmanBBox("bbox_xyah")
    states = {
        **bbox.state_entries(),
        "bbox_xyxy": State(
            schema=TensorSpec(shape=(4,), dtype=torch.float32),
            process=_ProjectXyahToXyxy(),
            observation=_ProjectXyahToXyxyAfterUpdate(),
            init=FromDetectionField("bbox_xyxy"),
        ),
    }
    return unitrack.Tracker(
        root=Pipe(
            cost=BoxIoU("bbox_xyxy"),
            assoc=Associate(Jonker(threshold=0.7)),
        ),
        states=states,
        lifecycle=StandardLifecycle(min_hits=min_hits, max_age=2),
        visibility=ConfirmedOnly(),
    )


def _two_box_detections(shift: float) -> Detections:
    return Detections(
        index=torch.arange(2, dtype=torch.int64),
        bbox_xyah=torch.tensor(
            [[10.0 + shift, 10.0, 0.5, 20.0], [30.0 + shift, 30.0, 0.5, 20.0]]
        ),
        bbox_xyxy=torch.tensor(
            [
                [5.0 + shift, 0.0, 15.0 + shift, 20.0],
                [25.0 + shift, 20.0, 35.0 + shift, 40.0],
            ]
        ),
        batch_size=[2],
    )


def test_sort_recipe_assigns_ids_across_two_frames():
    tr = _build_sort_tracker(min_hits=1)
    res0 = tr.step(
        tr.empty_snapshot(),
        _two_box_detections(0.0),
        FrameContext.make(0, delta=1.0),
        1,
    )
    res1 = tr.step(
        res0.snapshot,
        _two_box_detections(0.5),
        FrameContext.make(1, delta=1.0),
        res0.next_id,
    )
    assert res0.ids.tolist() == [1, 2]
    assert res1.ids.tolist() == [1, 2]
    assert res1.snapshot.id.tolist() == [1, 2]


def test_sort_recipe_confirmed_only_includes_first_frame_spawns():
    """min_hits=1 + ConfirmedOnly must surface newly-spawned tracklets at frame 0."""
    tr = _build_sort_tracker(min_hits=1)
    res = tr.step(
        tr.empty_snapshot(),
        _two_box_detections(0.0),
        FrameContext.make(0, delta=1.0),
        1,
    )
    # Before the fix, ConfirmedOnly returned an empty tensor because the
    # "matched_pairs" only referenced the pre-init snapshot.
    assert res.ids.numel() == 2
    assert res.ids.tolist() == [1, 2]


def test_sort_recipe_confirmed_only_hides_tentative_under_higher_min_hits():
    """min_hits=2 keeps brand-new tracklets Tentative on frame 0 → hidden."""
    tr = _build_sort_tracker(min_hits=2)
    res = tr.step(
        tr.empty_snapshot(),
        _two_box_detections(0.0),
        FrameContext.make(0, delta=1.0),
        1,
    )
    assert res.ids.numel() == 0
