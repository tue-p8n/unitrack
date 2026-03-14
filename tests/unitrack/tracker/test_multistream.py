# tests/unitrack/tracker/test_multistream.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import MultiStream, Tracker


def _build_tracker() -> Tracker:
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


def _ds(kernel: torch.Tensor) -> Detections:
    m = kernel.shape[0]
    return Detections(
        index=torch.arange(m, dtype=torch.int64), kernel=kernel, batch_size=[m]
    )


def test_streams_isolated_under_auto_fork():
    tr = _build_tracker()
    ms = MultiStream(tr)

    res_a = ms.step(
        stream_key=0,
        detections=_ds(torch.tensor([[1.0, 0.0]])),
        ctx=FrameContext.make(0, stream_key=0),
    )
    res_b = ms.step(
        stream_key=1,
        detections=_ds(torch.tensor([[0.0, 1.0]])),
        ctx=FrameContext.make(0, stream_key=1),
    )

    # Each stream gets its own ID counter starting at 1.
    assert res_a.ids.tolist() == [1]
    assert res_b.ids.tolist() == [1]


def test_reset_zeros_one_stream_only():
    tr = _build_tracker()
    ms = MultiStream(tr)
    ms.step(
        stream_key=0,
        detections=_ds(torch.tensor([[1.0, 0.0]])),
        ctx=FrameContext.make(0, stream_key=0),
    )
    ms.step(
        stream_key=1,
        detections=_ds(torch.tensor([[0.0, 1.0]])),
        ctx=FrameContext.make(0, stream_key=1),
    )
    ms.reset(0)
    assert ms.snapshot_of(0).batch_size[0] == 0
    assert ms.snapshot_of(1).batch_size[0] == 1


def test_reset_all_when_key_omitted():
    tr = _build_tracker()
    ms = MultiStream(tr)
    ms.step(
        stream_key=0,
        detections=_ds(torch.tensor([[1.0, 0.0]])),
        ctx=FrameContext.make(0, stream_key=0),
    )
    ms.step(
        stream_key=1,
        detections=_ds(torch.tensor([[0.0, 1.0]])),
        ctx=FrameContext.make(0, stream_key=1),
    )
    ms.reset()
    assert ms.snapshot_of(0).batch_size[0] == 0
    assert ms.snapshot_of(1).batch_size[0] == 0


def test_ordered_no_interleaving_allows_reopen_after_end_stream():
    """``end_stream(key)`` must drop *key* from ``OrderedNoInterleaving``'s seen
    set so callers can reopen the same key in a later session without hitting
    the "re-encountered after interleaving" check."""
    from unitrack.tracker import OrderedNoInterleaving

    tr = _build_tracker()
    ms = MultiStream(tr, fork_policy=OrderedNoInterleaving())

    ms.step(
        stream_key=0,
        detections=_ds(torch.tensor([[1.0, 0.0]])),
        ctx=FrameContext.make(0, stream_key=0),
    )
    ms.step(
        stream_key=1,
        detections=_ds(torch.tensor([[0.0, 1.0]])),
        ctx=FrameContext.make(0, stream_key=1),
    )
    # Without the end_stream hook, this would raise:
    #   ValueError: key 0 re-encountered after interleaving
    ms.end_stream(0)
    ms.step(
        stream_key=0,
        detections=_ds(torch.tensor([[1.0, 0.0]])),
        ctx=FrameContext.make(1, stream_key=0),
    )
