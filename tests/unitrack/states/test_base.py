from __future__ import annotations

import torch
from unitrack.data import Detections, FrameContext, MatchOutcome, TensorSpec, Tracklets
from unitrack.states.base import Initializer, Observation, Process, State


def test_state_construct_with_protocol_objects():
    class _NoProcess:
        def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
            """No-op process."""
            del ctx
            return cs

    class _ReplaceObs:
        def __call__(
            self,
            cs: Tracklets,
            ds: Detections,
            m: MatchOutcome,
            ctx: FrameContext,
        ) -> Tracklets:
            """No-op observation."""
            del ds, m, ctx
            return cs

    class _Init:
        def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
            """Return zeros."""
            del ctx
            return torch.zeros(ds.batch_size[0], 4)

    s = State(
        schema=TensorSpec(shape=(4,), dtype=torch.float32),
        process=_NoProcess(),
        observation=_ReplaceObs(),
        init=_Init(),
    )
    assert s.schema.shape == (4,)
    assert isinstance(s.process, Process)
    assert isinstance(s.observation, Observation)
    assert isinstance(s.init, Initializer)
