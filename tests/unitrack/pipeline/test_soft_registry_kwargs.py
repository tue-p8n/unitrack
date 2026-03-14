# tests/unitrack/pipeline/test_soft_registry_kwargs.py
"""Regression: default_soft_registry honours epsilon / temperature / sinkhorn_iters."""

from __future__ import annotations

from unitrack.assignment import Associate, Jonker, SoftAssignment
from unitrack.gates import MotionGate
from unitrack.gates.soft import SoftMotionGate
from unitrack.pipeline.diff import default_soft_registry, walk_swap


def test_custom_epsilon_propagates_through_associate_swap():
    reg = default_soft_registry(epsilon=0.42, sinkhorn_iters=7)
    swapped = walk_swap(Associate(Jonker(threshold=0.5)), reg)
    assert isinstance(swapped, Associate)
    inner = swapped.assignment
    assert isinstance(inner, SoftAssignment)
    assert inner.epsilon == 0.42
    assert inner.num_iter == 7


def test_custom_temperature_propagates_through_motion_gate_swap():
    reg = default_soft_registry(temperature=3.5)
    swapped = walk_swap(MotionGate("p", "p_cov", max_chi2=5.0), reg)
    assert isinstance(swapped, SoftMotionGate)
    assert swapped.temperature == 3.5


def test_defaults_unchanged_when_no_kwargs_passed():
    reg = default_soft_registry()
    swapped = walk_swap(Associate(Jonker(threshold=0.5)), reg)
    assert swapped.assignment.epsilon == 0.1
    assert swapped.assignment.num_iter == 50
