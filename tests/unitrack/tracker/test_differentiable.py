# tests/unitrack/tracker/test_differentiable.py
from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker, SoftAssignment
from unitrack.costs import Cosine
from unitrack.data import TensorSpec
from unitrack.gates import MotionGate
from unitrack.gates.soft import SoftMotionGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import Tracker


def test_differentiable_true_swaps_jonker_for_soft_assignment():
    pipe = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    tr = Tracker(
        root=pipe,
        states={
            "kernel": State(
                schema=TensorSpec(shape=(4,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
        differentiable=True,
    )
    # The root's assoc must now be SoftAssignment-backed.
    inner = tr.root.assoc.assignment
    assert isinstance(inner, SoftAssignment)


def test_differentiable_true_swaps_motion_gate_for_soft():
    inner = MotionGate("p", "p_cov", max_chi2=4.0)
    pipe = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    from unitrack.pipeline import Gated

    gated = Gated(gate=inner, then=pipe)

    tr = Tracker(
        root=gated,
        states={
            "kernel": State(
                schema=TensorSpec(shape=(4,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
        differentiable=True,
    )
    assert isinstance(tr.root.gate, SoftMotionGate)


def test_differentiable_walker_descends_into_sequential():
    # Regression for C1: ``Sequential`` is not a frozen dataclass, so the
    # walker's ``dataclasses.replace`` path used to raise on it.
    from unitrack.pipeline import Sequential

    pipe_a = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
    pipe_b = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.7)))
    seq = Sequential([pipe_a, pipe_b])

    tr = Tracker(
        root=seq,
        states={
            "kernel": State(
                schema=TensorSpec(shape=(4,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
        differentiable=True,
    )
    # Both inner Pipes had their assoc swapped to a SoftAssignment-backed Associate.
    assert all(
        isinstance(child.assoc.assignment, SoftAssignment) for child in tr.root.children
    )
