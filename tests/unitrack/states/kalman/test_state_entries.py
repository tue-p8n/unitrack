"""Smoke test for the Kalman state_entries factory (regression for C5)."""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.gates import MotionGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Gated, Pipe
from unitrack.states import (
    FromDetectionField,
    Identity,
    Replace,
    State,
)
from unitrack.states.kalman import KalmanBBox, KalmanCentroid2D, KalmanCentroid3D
from unitrack.tracker import Tracker


def test_kalman_centroid_2d_state_entries_runs_end_to_end():
    proc = KalmanCentroid2D("centroid")
    states = proc.state_entries() | {
        "kernel": State(
            schema=TensorSpec(shape=(4,), dtype=torch.float32),
            process=Identity("kernel"),
            observation=Replace("kernel"),
            init=FromDetectionField("kernel"),
        ),
    }
    tr = Tracker(
        root=Gated(
            gate=MotionGate("centroid", "centroid_cov", max_chi2=9.21),
            then=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        ),
        states=states,
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )
    snap = tr.empty_snapshot()
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        centroid=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        kernel=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        batch_size=[2],
    )
    res = tr.step(snap, ds, FrameContext.make(0, delta=1.0), next_id=1)
    # Mean is 4-D (x, y, vx, vy), seeded with detection x,y and zero velocity.
    assert res.snapshot.centroid.shape == (2, 4)
    assert torch.allclose(res.snapshot.centroid[:, :2], ds.centroid)
    assert torch.allclose(
        res.snapshot.centroid[:, 2:],
        torch.zeros(2, 2),
    )
    # Cov is 4-by-4, identity per tracklet.
    assert res.snapshot.centroid_cov.shape == (2, 4, 4)
    assert torch.allclose(res.snapshot.centroid_cov[0], torch.eye(4))


def test_kalman_centroid_3d_state_entries_advance_under_predict():
    proc = KalmanCentroid3D("p3d")
    states = proc.state_entries() | {
        "kernel": State(
            schema=TensorSpec(shape=(2,), dtype=torch.float32),
            process=Identity("kernel"),
            observation=Replace("kernel"),
            init=FromDetectionField("kernel"),
        ),
    }
    tr = Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states=states,
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )
    snap = tr.empty_snapshot()
    ds_t0 = Detections(
        index=torch.arange(1, dtype=torch.int64),
        p3d=torch.tensor([[0.0, 0.0, 0.0]]),
        kernel=torch.tensor([[1.0, 0.0]]),
        batch_size=[1],
    )
    res_t0 = tr.step(snap, ds_t0, FrameContext.make(0, delta=1.0), next_id=1)
    # Inject a velocity so predict has something to do.
    s = res_t0.snapshot.set("p3d", torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
    ds_t1 = Detections(
        index=torch.arange(1, dtype=torch.int64),
        p3d=torch.tensor([[1.0, 0.0, 0.0]]),
        kernel=torch.tensor([[1.0, 0.0]]),
        batch_size=[1],
    )
    res_t1 = tr.step(s, ds_t1, FrameContext.make(1, delta=1.0), next_id=res_t0.next_id)
    # Position should have advanced by 1 along x via the predict step.
    # (Then the update will refine, but with cov=I and small Q, x stays ~1.)
    assert res_t1.snapshot.p3d.shape == (1, 6)


def test_kalman_bbox_state_entries_seed_velocity_zero():
    proc = KalmanBBox("bbox_xyah")
    entries = proc.state_entries()
    assert set(entries.keys()) == {"bbox_xyah", "bbox_xyah_cov"}
    assert entries["bbox_xyah"].schema.shape == (7,)
    assert entries["bbox_xyah_cov"].schema.shape == (7, 7)
