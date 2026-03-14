from __future__ import annotations

import pytest
import torch
from unitrack.data import FrameContext, Tracklets
from unitrack.states.kalman import KalmanBBox


def _make_snap(state_dim: int, mean: torch.Tensor) -> Tracklets:
    n = mean.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        bbox=mean,
        bbox_cov=torch.eye(state_dim).expand(n, state_dim, state_dim).contiguous(),
        batch_size=[n],
    )


def test_kalman_bbox_predict_advances_state():
    # x,y,a,h,vx,vy,va in 7-D SORT model
    mean = torch.tensor([[100.0, 200.0, 1.0, 50.0, 1.0, 0.0, 0.0]])
    snap = _make_snap(7, mean)
    out = KalmanBBox()(snap, FrameContext.make(0, delta=1.0))
    # bbox center moves by velocity; vy=0 so y unchanged
    assert torch.allclose(out.bbox[0, :2], torch.tensor([101.0, 200.0]))


def test_kalman_bbox_sort_state_is_7d():
    """SORT model's default state dimensionality is 7."""
    bb = KalmanBBox(model="sort")
    assert bb.state_dim == 7
    entries = bb.state_entries()
    assert entries["bbox"].schema.shape == (7,)


def test_kalman_bbox_deepsort_state_is_8d():
    """DeepSORT model adds a height-velocity component, total 8 dims."""
    bb = KalmanBBox(model="deepsort")
    assert bb.state_dim == 8
    entries = bb.state_entries()
    assert entries["bbox"].schema.shape == (8,)
    assert entries["bbox_cov"].schema.shape == (8, 8)


def test_kalman_bbox_deepsort_predict_uses_height_velocity():
    """DeepSORT's height-velocity `vh` advances `h` over dt — the key
    deviation from SORT, where `h` is observed but not predicted."""
    # 8-D state: x, y, a, h, vx, vy, va, vh
    mean = torch.tensor([[100.0, 200.0, 1.0, 50.0, 0.0, 0.0, 0.0, 2.0]])
    snap = _make_snap(8, mean)
    out = KalmanBBox(model="deepsort")(snap, FrameContext.make(0, delta=1.0))
    # h advances by vh=2.0 over dt=1.0; other coords unchanged (all velocities 0).
    assert torch.allclose(out.bbox[0, 3], torch.tensor(52.0))
    assert torch.allclose(out.bbox[0, :3], torch.tensor([100.0, 200.0, 1.0]))


def test_kalman_bbox_rejects_unknown_model():
    """Unknown model strings fail at construction with a clear message."""
    with pytest.raises(ValueError, match="must be 'sort' or 'deepsort'"):
        KalmanBBox(model="unknown")  # type: ignore[arg-type]


def test_kalman_bbox_sort_h_not_predicted():
    """SORT's `h` is *not* predicted (no `vh` velocity in state). One
    predict step with the velocity components zeroed must leave h
    untouched, distinguishing SORT from DeepSORT."""
    # 7-D state with all velocities zero
    mean = torch.tensor([[100.0, 200.0, 1.0, 50.0, 0.0, 0.0, 0.0]])
    snap = _make_snap(7, mean)
    out = KalmanBBox(model="sort")(snap, FrameContext.make(0, delta=1.0))
    # h should be unchanged because SORT has no vh
    assert torch.allclose(out.bbox[0, 3], torch.tensor(50.0))
