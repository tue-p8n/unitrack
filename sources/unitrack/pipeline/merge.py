"""
Merge strategies for :class:`~unitrack.pipeline.combinators.Parallel`.

Each strategy declares an explicit policy for how branch gates combine.
:class:`WeightedSum` and :class:`Mean` AND-combine (conservative): a
pair is allowed only when every branch's gate allows it. :class:`Min`
and :class:`Max` OR-combine (permissive): a pair is allowed when any
branch allows it; this matches the "best-of-branches" reduction
semantics. :class:`StackReduce` defaults to AND for user-supplied
reducers; pass ``gate_mode="or"`` when the reducer is best-of-branches.
Biases always sum additively regardless of mode.
"""

from __future__ import annotations

import dataclasses
import typing

import torch

from unitrack.data import CostExpression

__all__ = ["Max", "Mean", "Merge", "Min", "StackReduce", "WeightedSum"]

GateCombineMode = typing.Literal["and", "or"]


@typing.runtime_checkable
class Merge(typing.Protocol):
    """Protocol for strategies that combine a list of cost expressions."""

    def __call__(self, exprs: list[CostExpression]) -> CostExpression:
        """
        Merge ``exprs`` into one :class:`~unitrack.data.CostExpression`.

        Parameters
        ----------
        exprs
            Per-branch cost expressions, all of shape ``(N, M)``.

        Returns
        -------
        CostExpression
            The reduced expression.

        """
        ...


def _stack(exprs: list[CostExpression]) -> torch.Tensor:
    return torch.stack([e.matrix for e in exprs])


def _combine_attached(
    exprs: list[CostExpression], *, gate_mode: GateCombineMode
) -> dict:
    """
    Fold gate and bias slots across branches according to ``gate_mode``.

    In ``"and"`` mode (conservative), masks are conjoined where both
    sides have one and a single-side mask is promoted unchanged (the
    absent branch is treated as ``True`` everywhere). In ``"or"`` mode
    (permissive), masks are disjoined when every branch carries one and
    the result collapses to ``None`` whenever any branch lacks a gate.
    Biases always sum additively.
    """
    return {
        "gate_pair": _fold_mask([e.gate_pair for e in exprs], gate_mode=gate_mode),
        "gate_cs": _fold_mask([e.gate_cs for e in exprs], gate_mode=gate_mode),
        "gate_ds": _fold_mask([e.gate_ds for e in exprs], gate_mode=gate_mode),
        "bias": _fold_bias([e.bias for e in exprs]),
    }


def _fold_mask(
    masks: list[torch.Tensor | None], *, gate_mode: GateCombineMode
) -> torch.Tensor | None:
    """AND- or OR-fold a list of optional boolean masks per ``gate_mode``."""
    if gate_mode == "and":
        acc: torch.Tensor | None = None
        for m in masks:
            if m is None:
                continue  # "no gate" = all True; AND-identity
            acc = m if acc is None else acc & m
        return acc
    # OR mode: if any branch has no gate, the union is "all True" → None.
    if any(m is None for m in masks):
        return None
    if not masks:
        return None
    acc = masks[0]
    for m in masks[1:]:
        acc = acc | m  # type: ignore[union-attr]
    return acc


def _fold_bias(biases: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Sum non-None biases additively."""
    acc: torch.Tensor | None = None
    for b in biases:
        if b is None:
            continue
        acc = b if acc is None else acc + b
    return acc


@dataclasses.dataclass(frozen=True, slots=True)
class WeightedSum:
    """
    Merge by weighted sum of cost matrices; AND-combines gates.

    Attributes
    ----------
    weights : list of float
        One weight per branch; length must match the number of inputs.

    """

    weights: list[float]

    def __call__(self, exprs: list[CostExpression]) -> CostExpression:
        """
        Return the weighted-sum cost expression.

        Parameters
        ----------
        exprs
            Per-branch cost expressions.

        Returns
        -------
        CostExpression
            ``(N, M)`` weighted-sum matrix with AND-combined gates.

        Raises
        ------
        ValueError
            If ``len(weights) != len(exprs)``.

        """
        if len(self.weights) != len(exprs):
            msg = f"WeightedSum: got {len(exprs)} branches, {len(self.weights)} weights"
            raise ValueError(msg)
        stacked = _stack(exprs)
        w = torch.tensor(self.weights, dtype=stacked.dtype, device=stacked.device)
        m = (stacked * w[:, None, None]).sum(dim=0)
        return CostExpression.from_matrix(
            m, **_combine_attached(exprs, gate_mode="and")
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Min:
    """
    Merge by elementwise minimum of cost matrices; OR-combines gates.

    Notes
    -----
    OR-combine matches the best-of-branches reduction: a pair gated by
    one branch but allowed by another is surfaced through the
    permissive union.

    """

    def __call__(self, exprs: list[CostExpression]) -> CostExpression:
        """
        Return the elementwise-minimum cost expression.

        Returns
        -------
        CostExpression
            ``(N, M)`` minimum matrix with OR-combined gates.

        """
        m = _stack(exprs).min(dim=0).values
        return CostExpression.from_matrix(m, **_combine_attached(exprs, gate_mode="or"))


@dataclasses.dataclass(frozen=True, slots=True)
class Max:
    """
    Merge by elementwise maximum of cost matrices; OR-combines gates.

    See :class:`Min` for the gate rationale; :class:`Max` likewise
    represents a best-of-branches worst-case bound.
    """

    def __call__(self, exprs: list[CostExpression]) -> CostExpression:
        """
        Return the elementwise-maximum cost expression.

        Returns
        -------
        CostExpression
            ``(N, M)`` maximum matrix with OR-combined gates.

        """
        m = _stack(exprs).max(dim=0).values
        return CostExpression.from_matrix(m, **_combine_attached(exprs, gate_mode="or"))


@dataclasses.dataclass(frozen=True, slots=True)
class Mean:
    """Merge by elementwise mean of cost matrices; AND-combines gates."""

    def __call__(self, exprs: list[CostExpression]) -> CostExpression:
        """
        Return the elementwise-mean cost expression.

        Returns
        -------
        CostExpression
            ``(N, M)`` mean matrix with AND-combined gates.

        """
        m = _stack(exprs).mean(dim=0)
        return CostExpression.from_matrix(
            m, **_combine_attached(exprs, gate_mode="and")
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StackReduce:
    """
    Apply a user-supplied reducer to stacked cost matrices.

    Attributes
    ----------
    reducer : ~collections.abc.Callable
        Function from a ``(K, N, M)`` tensor to an ``(N, M)`` tensor.
    gate_mode : {'and', 'or'}
        Gate-combination policy. Defaults to ``'and'`` (conservative);
        pass ``'or'`` when the reducer is best-of-branches.

    """

    reducer: typing.Callable[[torch.Tensor], torch.Tensor]
    gate_mode: GateCombineMode = "and"

    def __call__(self, exprs: list[CostExpression]) -> CostExpression:
        """
        Return the reducer's output as a cost expression.

        Returns
        -------
        CostExpression
            ``(N, M)`` matrix from :attr:`reducer` with gates combined
            according to :attr:`gate_mode`.

        """
        m = self.reducer(_stack(exprs))
        return CostExpression.from_matrix(
            m, **_combine_attached(exprs, gate_mode=self.gate_mode)
        )
