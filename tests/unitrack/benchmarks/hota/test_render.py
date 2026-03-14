import torch
from unitrack.benchmarks.hota.render import TrackRemap, render_pred_panoptic


def test_render_encodes_semantic_offset_plus_per_class_instance():
    h, w, offset = 4, 4, 1000
    remap = TrackRemap(offset=offset)
    m0 = torch.zeros((h, w), dtype=torch.bool)
    m0[0:2, :] = True
    m1 = torch.zeros((h, w), dtype=torch.bool)
    m1[2:4, :] = True
    pan = render_pred_panoptic(
        masks=torch.stack([m0, m1]),
        categories=torch.tensor([13, 11]),
        track_ids=torch.tensor([7, 4]),  # different classes -> each instance 1
        height=h,
        width=w,
        offset=offset,
        remap=remap,
    )
    # Indexing is per-class, so both tracks get instance 1 within their class.
    assert pan[0, 0] == 13 * offset + 1  # track 7 (class 13) -> instance 1
    assert pan[2, 0] == 11 * offset + 1  # track 4 (class 11) -> instance 1
    # stability across frames: same track id keeps its instance index
    pan2 = render_pred_panoptic(
        masks=m0[None],
        categories=torch.tensor([13]),
        track_ids=torch.tensor([7]),
        height=h,
        width=w,
        offset=offset,
        remap=remap,
    )
    assert pan2[0, 0] == 13 * offset + 1


def test_render_same_class_tracks_get_distinct_indices():
    h, w, offset = 4, 4, 1000
    remap = TrackRemap(offset=offset)
    m0 = torch.zeros((h, w), dtype=torch.bool)
    m0[0:2, :] = True
    m1 = torch.zeros((h, w), dtype=torch.bool)
    m1[2:4, :] = True
    pan = render_pred_panoptic(
        masks=torch.stack([m0, m1]),
        categories=torch.tensor([11, 11]),  # same class
        track_ids=torch.tensor([7, 4]),
        height=h,
        width=w,
        offset=offset,
        remap=remap,
    )
    assert pan[0, 0] == 11 * offset + 1  # track 7 -> instance 1
    assert pan[2, 0] == 11 * offset + 2  # track 4 -> instance 2


def test_render_skips_gated_out_detections():
    h = w = 4
    remap = TrackRemap(offset=1000)
    m = torch.ones((1, h, w), dtype=torch.bool)
    pan = render_pred_panoptic(
        masks=m,
        categories=torch.tensor([11]),
        track_ids=torch.tensor([-1]),
        height=h,
        width=w,
        offset=1000,
        remap=remap,
    )
    assert (pan == 0).all()
