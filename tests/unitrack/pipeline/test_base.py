from __future__ import annotations

import torch
from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    Gate,
    MatchOutcome,
    Tracklets,
)
from unitrack.pipeline.base import (
    Associator,
    CostProducer,
    GateProducer,
    PipelineTypeError,
)


def test_stage_protocols_are_runtime_checkable():
    class _M:
        def __call__(
            self,
            cs: Tracklets,
            ds: Detections,
            ctx: FrameContext,
            cost: CostExpression | None = None,
        ) -> MatchOutcome:
            del cs, ds, ctx, cost
            return MatchOutcome.empty()

    class _C:
        def __call__(
            self, cs: Tracklets, ds: Detections, ctx: FrameContext
        ) -> CostExpression:
            del cs, ds, ctx
            return CostExpression.from_matrix(torch.zeros(0, 0))

    class _G:
        def __call__(self, cs: Tracklets, ds: Detections, ctx: FrameContext) -> Gate:
            del cs, ds, ctx
            return Gate.PerPair(mask=torch.zeros(0, 0, dtype=torch.bool))

    assert isinstance(_M(), Associator)
    assert isinstance(_C(), CostProducer)
    assert isinstance(_G(), GateProducer)


def test_pipeline_type_error_carries_path():
    err = PipelineTypeError("expected CostProducer", path=["root", "children", "0"])
    assert "root.children.0" in str(err)
