# tests/unitrack/tracker/test_batch_vmap_conformance.py
"""C8 conformance: batched-solve fast path matches the loop path bit-for-bit."""

from __future__ import annotations

import pytest
import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import BatchTracker, Tracker


def _build():
    return Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states={
            "kernel": State(
                schema=TensorSpec(shape=(2,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )


def _ds(kernel):
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_batched_fast_path_matches_loop_path():
    """Same inputs, same per-slot outputs whether the fast path or loop runs."""
    torch.manual_seed(7)

    # Frame 0: spawn 2 tracklets per slot (loop path — fresh memories).
    # Frame 1: 2 detections per slot, both N>0 and M>0 — fast path engages.
    frames = [
        (
            _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
            _ds(torch.tensor([[0.6, 0.8], [0.8, 0.6]])),
        ),
        (
            _ds(torch.tensor([[0.95, 0.05], [0.05, 0.95]])),
            _ds(torch.tensor([[0.5, 0.85], [0.85, 0.5]])),
        ),
    ]
    ctxs = [
        (FrameContext.make(0, stream_key=0), FrameContext.make(0, stream_key=1)),
        (FrameContext.make(1, stream_key=0), FrameContext.make(1, stream_key=1)),
    ]

    # Batched path.
    tr_b = _build()
    bt = BatchTracker(tr_b, batch_size=2)
    batched_results = []
    for (ds_a, ds_b), (ctx_a, ctx_b) in zip(frames, ctxs, strict=True):
        batched_results.append(bt.step([ds_a, ds_b], [ctx_a, ctx_b]))

    # Sequential reference path, per slot.
    seq_results = [[], []]
    for slot in (0, 1):
        tr = _build()
        snap = tr.empty_snapshot()
        next_id = 1
        for fi in range(len(frames)):
            res = tr.step(snap, frames[fi][slot], ctxs[fi][slot], next_id)
            snap = res.snapshot
            next_id = res.next_id
            seq_results[slot].append(res)

    # On frame 0 the fast path doesn't engage (snapshots empty) — both paths
    # are the loop path. Frame 1 has N=2, M=2, NoLifecycle — fast path engages.
    for fi in range(len(frames)):
        for slot in (0, 1):
            r_batched = batched_results[fi][slot]
            r_seq = seq_results[slot][fi]
            assert torch.equal(r_batched.snapshot.id, r_seq.snapshot.id), (
                f"frame={fi} slot={slot} ids diverge"
            )
            assert torch.equal(
                r_batched.match.matched_pairs, r_seq.match.matched_pairs
            ), f"frame={fi} slot={slot} pairs diverge"
            assert torch.allclose(
                r_batched.snapshot["kernel"], r_seq.snapshot["kernel"], atol=1e-6
            ), f"frame={fi} slot={slot} kernel fields diverge"


@pytest.mark.parametrize(
    ("n", "m"),
    [(1, 1), (1, 3), (3, 1), (5, 5), (10, 2)],
    ids=["1x1", "1x3", "3x1", "5x5", "10x2"],
)
def test_batched_fast_path_matches_loop_path_param(n: int, m: int):
    """Fast path vs loop path equivalence across square / ragged / single-row shapes."""
    torch.manual_seed(13 + n * 7 + m)

    # Spawn N tracklets per slot with kernels around a per-slot anchor.
    anchor_a = torch.randn(n, 2)
    anchor_b = torch.randn(n, 2)
    frame0 = (_ds(anchor_a), _ds(anchor_b))
    # Frame 1 has M detections (possibly != N) — fast path engages because the
    # snapshots have equal N across slots and NoLifecycle is in effect.
    frame1 = (
        _ds(anchor_a[:m] if m <= n else torch.cat([anchor_a, torch.randn(m - n, 2)])),
        _ds(anchor_b[:m] if m <= n else torch.cat([anchor_b, torch.randn(m - n, 2)])),
    )

    frames = [frame0, frame1]
    ctxs = [
        (FrameContext.make(0, stream_key=0), FrameContext.make(0, stream_key=1)),
        (FrameContext.make(1, stream_key=0), FrameContext.make(1, stream_key=1)),
    ]

    tr_b = _build()
    bt = BatchTracker(tr_b, batch_size=2)
    batched_results = []
    for (ds_a, ds_b), (ctx_a, ctx_b) in zip(frames, ctxs, strict=True):
        batched_results.append(bt.step([ds_a, ds_b], [ctx_a, ctx_b]))

    seq_results = [[], []]
    for slot in (0, 1):
        tr = _build()
        snap = tr.empty_snapshot()
        next_id = 1
        for fi in range(len(frames)):
            res = tr.step(snap, frames[fi][slot], ctxs[fi][slot], next_id)
            snap = res.snapshot
            next_id = res.next_id
            seq_results[slot].append(res)

    for fi in range(len(frames)):
        for slot in (0, 1):
            r_b = batched_results[fi][slot]
            r_s = seq_results[slot][fi]
            assert torch.equal(r_b.snapshot.id, r_s.snapshot.id), (
                f"shape=({n},{m}) frame={fi} slot={slot} ids diverge"
            )
            assert torch.equal(r_b.match.matched_pairs, r_s.match.matched_pairs), (
                f"shape=({n},{m}) frame={fi} slot={slot} pairs diverge"
            )
            assert torch.allclose(
                r_b.snapshot["kernel"], r_s.snapshot["kernel"], atol=1e-6
            ), f"shape=({n},{m}) frame={fi} slot={slot} kernel diverges"


def test_batched_fast_path_skipped_for_empty_first_frame():
    """First frame's snapshots are empty — must use loop path; results still match."""
    tr = _build()
    bt = BatchTracker(tr, batch_size=2)
    dets = [_ds(torch.tensor([[1.0, 0.0]])), _ds(torch.tensor([[0.0, 1.0]]))]
    ctxs = [FrameContext.make(0, stream_key=0), FrameContext.make(0, stream_key=1)]
    # is_vmap_safe says yes (NoLifecycle + uniform), but _can_dispatch_batched
    # must say no because per-slot snapshot N=0.
    assert bt.is_vmap_safe(dets) is True
    assert bt._can_dispatch_batched(dets) is False
    # Still runs correctly via loop path.
    results = bt.step(dets, ctxs)
    assert len(results) == 2
    assert sorted(results[0].ids.tolist()) == [1]
    assert sorted(results[1].ids.tolist()) == [1]
