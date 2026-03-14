"""Unitrack — image-trained video tracking primitives."""

from __future__ import annotations

from . import assignment, costs, data, gates, lifecycle, pipeline, states
from .data import (
    ClipDetections,
    ClipFrameContext,
    ClipMatchOutcome,
    ClipTracklets,
    CostExpression,
    Detections,
    FrameContext,
    Gate,
    MatchOutcome,
    TensorSpec,
    Tracklets,
)
from .lifecycle import (
    ConfirmedOnly,
    IncludeAll,
    IncludeTentative,
    NoLifecycle,
    StandardLifecycle,
    TrackletStatus,
)
from .pipeline import (
    Associator,
    CostProducer,
    Filter,
    Gated,
    GateProducer,
    Iterate,
    Parallel,
    Pipe,
    PipelineTypeError,
    Sequential,
    Stage,
)
from .tracker import (
    AutoForkOnNewKey,
    BatchTracker,
    ClipTracker,
    ForkPolicy,
    MultiStream,
    OrderedNoInterleaving,
    StepResult,
    Tracker,
    TrackletMemory,
)

__all__ = [
    "Associator",
    "AutoForkOnNewKey",
    "BatchTracker",
    "ClipDetections",
    "ClipFrameContext",
    "ClipMatchOutcome",
    "ClipTracker",
    "ClipTracklets",
    "ConfirmedOnly",
    "CostExpression",
    "CostProducer",
    "Detections",
    "Filter",
    "ForkPolicy",
    "FrameContext",
    "Gate",
    "GateProducer",
    "Gated",
    "IncludeAll",
    "IncludeTentative",
    "Iterate",
    "MatchOutcome",
    "MultiStream",
    "NoLifecycle",
    "OrderedNoInterleaving",
    "Parallel",
    "Pipe",
    "PipelineTypeError",
    "Sequential",
    "Stage",
    "StandardLifecycle",
    "StepResult",
    "TensorSpec",
    "Tracker",
    "TrackletMemory",
    "TrackletStatus",
    "Tracklets",
    "assignment",
    "costs",
    "data",
    "gates",
    "lifecycle",
    "pipeline",
    "states",
]
__version__: str


def __getattr__(name: str):
    if name == "__version__":
        import importlib.metadata

        return importlib.metadata.version("unitrack")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return __all__
