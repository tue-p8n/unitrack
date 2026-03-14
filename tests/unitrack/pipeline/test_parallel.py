# tests/unitrack/pipeline/test_parallel.py
from __future__ import annotations

import pytest
import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import CDist, Cosine
from unitrack.data import Detections, FrameContext, Tracklets
from unitrack.pipeline import Parallel, Pipe
from unitrack.pipeline.base import PipelineTypeError
from unitrack.pipeline.merge import WeightedSum


def _two_tracklets(kernel, position):
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
        position=position,
        batch_size=[n],
    )


def test_parallel_weighted_sum_combines_appearance_and_geometry():
    cs_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ds_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cs_p = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    ds_p = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    cs = _two_tracklets(cs_k, cs_p)
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=ds_k,
        position=ds_p,
        batch_size=[2],
    )
    parallel = Parallel(
        children=[Cosine("kernel"), CDist("position")],
        merge=WeightedSum([1.0, 0.1]),
    )
    pipe = Pipe(cost=parallel, assoc=Associate(Jonker(threshold=2.0)))
    out = pipe(cs, ds, FrameContext.make(0))
    pairs = sorted(map(tuple, out.matched_pairs.tolist()))
    assert pairs == [(0, 0), (1, 1)]


def test_parallel_rejects_non_cost_producer_child():
    with pytest.raises(PipelineTypeError, match="CostProducer"):
        Parallel(
            children=[Cosine("kernel"), "not_a_stage"], merge=WeightedSum([1.0, 0.5])
        )
