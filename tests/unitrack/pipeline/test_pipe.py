# tests/unitrack/pipeline/test_pipe.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.pipeline import Pipe


def _cs(kernel: torch.Tensor) -> Tracklets:
    n = kernel.shape[0]
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
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


def test_pipe_runs_cost_then_associate_end_to_end():
    cs = _cs(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    ds = _ds(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    ctx = FrameContext.make(0, delta=0.0)

    pipe = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    out = pipe(cs, ds, ctx)
    pairs = sorted(map(tuple, out.matched_pairs.tolist()))
    assert pairs == [(0, 0), (1, 1)]


def test_pipe_construction_rejects_non_cost_producer():
    import pytest
    from unitrack.pipeline.base import PipelineTypeError

    class _Bogus:
        pass

    with pytest.raises(PipelineTypeError, match="cost"):
        Pipe(cost=_Bogus(), assoc=Associate(Jonker(threshold=0.5)))


def test_pipe_construction_rejects_non_associator():
    import pytest
    from unitrack.pipeline.base import PipelineTypeError

    with pytest.raises(PipelineTypeError, match="assoc"):
        Pipe(cost=Cosine("kernel"), assoc=Cosine("kernel"))
