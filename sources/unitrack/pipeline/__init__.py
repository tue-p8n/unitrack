"""
Stage protocols, combinators, merges, and the hard-to-soft tree rewrite.

Stages compose into a tree consumed by the
:class:`~unitrack.tracker.Tracker`. The protocols (:class:`Stage`,
:class:`~unitrack.pipeline.CostProducer`, :class:`GateProducer`,
:class:`~unitrack.pipeline.Associator`, :class:`~unitrack.pipeline.Lifecycle`,
:class:`~unitrack.pipeline.Visibility`) define the contract; the
combinators (:class:`Pipe`, :class:`Sequential`, :class:`Parallel`,
:class:`Gated`, :class:`Filter`, :class:`Iterate`) wire stages together.
"""

from __future__ import annotations

from .base import (
    Associator,
    CostProducer,
    GateProducer,
    Lifecycle,
    PipelineTypeError,
    Stage,
    Visibility,
)
from .combinators import Filter, Gated, Iterate, Parallel, Pipe, Sequential
from .diff import SoftRegistry, default_soft_registry, walk_swap
from .merge import Max, Mean, Merge, Min, StackReduce, WeightedSum

__all__ = [
    "Associator",
    "CostProducer",
    "Filter",
    "GateProducer",
    "Gated",
    "Iterate",
    "Lifecycle",
    "Max",
    "Mean",
    "Merge",
    "Min",
    "Parallel",
    "Pipe",
    "PipelineTypeError",
    "Sequential",
    "SoftRegistry",
    "StackReduce",
    "Stage",
    "Visibility",
    "WeightedSum",
    "default_soft_registry",
    "walk_swap",
]
