from __future__ import annotations

import torch
from unitrack.costs import GalleryCost
from unitrack.data import Detections, FrameContext, Tracklets


def _cs(gallery: torch.Tensor, count: torch.Tensor) -> Tracklets:
    n = gallery.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        emb=gallery[:, 0, :],
        emb_gallery=gallery,
        emb_count=count,
        batch_size=[n],
    )


def _ds(emb: torch.Tensor) -> Detections:
    return Detections(
        index=torch.arange(emb.shape[0]), emb=emb, batch_size=[emb.shape[0]]
    )


def test_shape_and_perfect_match_is_zero_cost():
    gallery = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # (1, 2, 3)
    cs = _cs(gallery, torch.tensor([2]))
    ds = _ds(torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))  # (2, 3)
    cost = GalleryCost("emb_gallery", "emb_count", "emb", reduce="max")(
        cs, ds, FrameContext.make(0)
    )
    assert cost.matrix.shape == (1, 2)
    assert torch.allclose(
        cost.matrix[0, 0], torch.tensor(0.0), atol=1e-6
    )  # query == slot 1
    assert cost.matrix[0, 1] > 0.9  # orthogonal to every slot


def test_max_reduce_beats_mean_when_one_view_matches():
    # gallery holds two dissimilar views; query equals the second.
    gallery = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    cs = _cs(gallery, torch.tensor([2]))
    ds = _ds(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    ctx = FrameContext.make(0)
    cmax = GalleryCost("emb_gallery", "emb_count", "emb", reduce="max")(
        cs, ds, ctx
    ).matrix
    cmean = GalleryCost("emb_gallery", "emb_count", "emb", reduce="mean")(
        cs, ds, ctx
    ).matrix
    assert cmax[0, 0] < cmean[0, 0]  # best-view (max) ignores the mismatching slot
    assert torch.allclose(cmax[0, 0], torch.tensor(0.0), atol=1e-6)


def test_empty_slots_are_masked():
    # capacity 3 but only slot 0 filled (count = 1); the zero slots must not match.
    gallery = torch.zeros(1, 3, 3)
    gallery[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    cs = _cs(gallery, torch.tensor([1]))
    ds = _ds(torch.tensor([[1.0, 0.0, 0.0]]))  # matches the only valid slot
    cost = GalleryCost("emb_gallery", "emb_count", "emb", reduce="mean")(
        cs, ds, FrameContext.make(0)
    )
    # mean over *valid* slots (just slot 0) is a perfect match, not diluted by empties.
    assert torch.allclose(cost.matrix[0, 0], torch.tensor(0.0), atol=1e-6)
