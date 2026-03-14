from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states import GalleryAppend, GalleryInitializer, gallery_state_entries


def _t(emb, gallery, count) -> Tracklets:
    n = emb.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        emb=emb,
        emb_gallery=gallery,
        emb_count=count,
        batch_size=[n],
    )


def _match_one() -> MatchOutcome:
    return MatchOutcome(
        matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
        tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
        detections_residual_index=torch.zeros(0, dtype=torch.int64),
        per_match_cost=torch.zeros(1),
        batch_size=[],
    )


def _seed(z, k):
    d = z.shape[1]
    ds = Detections(index=torch.tensor([0]), emb=z, batch_size=[1])
    gallery = GalleryInitializer("emb", k, d)(ds, FrameContext.make(0))
    return _t(z.clone(), gallery, torch.tensor([1]))


def test_initializer_seeds_first_slot():
    z = torch.tensor([[1.0, 2.0, 3.0]])
    cs = _seed(z, 4)
    assert torch.allclose(cs.emb_gallery[0, 0], z[0])
    assert torch.allclose(cs.emb_gallery[0, 1:], torch.zeros(3, 3))
    assert cs.emb_count[0] == 1


def test_append_evicts_oldest_in_ring_order():
    cs = _seed(torch.tensor([[1.0, 0.0, 0.0]]), 3)  # slot0 filled, count=1
    appender = GalleryAppend("emb", "emb")
    for v in ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [9.0, 9.0, 9.0]):
        ds = Detections(index=torch.tensor([0]), emb=torch.tensor([v]), batch_size=[1])
        cs = appender(cs, ds, _match_one(), FrameContext.make(0))
    # appends land in slots 1, 2, 0 (count goes 1->2->3->4).
    assert cs.emb_count[0] == 4
    assert torch.allclose(
        cs.emb_gallery[0, 0], torch.tensor([9.0, 9.0, 9.0])
    )  # overwrote slot 0
    assert torch.allclose(cs.emb_gallery[0, 1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(
        cs.emb[0], torch.tensor([9.0, 9.0, 9.0])
    )  # primary = most recent


def test_unmatched_tracklet_gallery_unchanged():
    # two tracklets; only tracklet 0 is in the matched pairs.
    two = Tracklets(
        id=torch.arange(2, dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.ones(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        emb=torch.tensor([[1.0, 0.0], [5.0, 5.0]]),
        emb_gallery=torch.zeros(2, 3, 2),
        emb_count=torch.tensor([1, 1]),
        batch_size=[2],
    )
    ds = Detections(
        index=torch.tensor([0]), emb=torch.tensor([[7.0, 7.0]]), batch_size=[1]
    )
    out = GalleryAppend("emb", "emb")(two, ds, _match_one(), FrameContext.make(0))
    assert out.emb_count[1] == 1  # tracklet 1 untouched
    assert torch.allclose(out.emb[1], torch.tensor([5.0, 5.0]))


def test_state_entries_keys():
    entries = gallery_state_entries("emb", dim=4, capacity=5)
    assert set(entries) == {"emb", "emb_gallery", "emb_count"}
