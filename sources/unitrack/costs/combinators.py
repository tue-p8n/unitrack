"""Cost-side combinators: :class:`Weighted`, :class:`Reduce`, :class:`Sinkhorn`."""

from __future__ import annotations

import dataclasses
import enum
import typing

import torch

from unitrack.assignment._soft import sinkhorn_log_plan as _sinkhorn_log_plan
from unitrack.data import CostExpression, Detections, FrameContext, Tracklets

__all__ = ["Reduce", "Reduction", "Sinkhorn", "Weighted"]


@dataclasses.dataclass(frozen=True, slots=True)
class Weighted:
    """
    Scale a child cost producer's output by a scalar weight.

    Attributes
    ----------
    inner : ~unitrack.pipeline.CostProducer
        Wrapped cost producer.
    weight : float
        Scalar multiplier applied to ``inner.matrix``.

    """

    inner: typing.Any
    weight: float

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Run :attr:`inner` and scale its cost matrix.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to :attr:`inner` unchanged.

        Returns
        -------
        CostExpression
            The inner expression with ``matrix`` multiplied by
            :attr:`weight`; attached gates and bias are preserved.

        """
        expr = self.inner(cs, ds, ctx)
        return CostExpression.from_matrix(
            self.weight * expr.matrix,
            gate_pair=expr.gate_pair,
            gate_cs=expr.gate_cs,
            gate_ds=expr.gate_ds,
            bias=expr.bias,
        )


class Reduction(enum.StrEnum):
    """
    Reduction method for combining multiple cost matrices.

    Notes
    -----
    A ``PRODUCT`` reduction was removed in 2.0. The elementwise product
    of cost matrices collapses to zero whenever any child cost is
    exactly zero, silently overriding every other term, and it is
    dimensionally suspicious (cost-by-cost). Use ``SUM`` with
    :class:`Weighted` children for the additive case, or ``MIN`` /
    ``MAX`` for the bound case.

    """

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"


@dataclasses.dataclass(frozen=True, slots=True)
class Reduce:
    """
    Combine multiple child cost producers via a reduction.

    Attributes
    ----------
    children : list of ~unitrack.pipeline.CostProducer
        Branches whose outputs are stacked and reduced.
    method : Reduction or str
        Reduction selector. String values are coerced to
        :class:`Reduction`.

    """

    children: list
    method: Reduction | str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Run every child and combine their cost matrices.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to each child unchanged.

        Returns
        -------
        CostExpression
            ``(N, M)`` reduced cost matrix. Per-child gate masks are
            AND-combined pointwise; biases sum additively.

        Raises
        ------
        ValueError
            If :attr:`method` is not a recognised :class:`Reduction`.

        """
        method = Reduction(self.method)
        exprs = [c(cs, ds, ctx) for c in self.children]
        stacked = torch.stack([e.matrix for e in exprs])
        match method:
            case Reduction.SUM:
                m = stacked.sum(dim=0)
            case Reduction.MEAN:
                m = stacked.mean(dim=0)
            case Reduction.MIN:
                m = stacked.min(dim=0).values
            case Reduction.MAX:
                m = stacked.max(dim=0).values
            case _:  # pyright: ignore[reportUnreachable]
                msg = f"Reduce: unknown reduction method {method!r}"  # pyright: ignore[reportUnreachable]
                raise ValueError(msg)
        # Combine attached gates pointwise (AND for masks; sum for biases).
        gate_pair = None
        gate_cs = None
        gate_ds = None
        bias = None
        for e in exprs:
            gate_pair = _and(gate_pair, e.gate_pair)
            gate_cs = _and(gate_cs, e.gate_cs)
            gate_ds = _and(gate_ds, e.gate_ds)
            bias = _add(bias, e.bias)
        return CostExpression.from_matrix(
            m,
            gate_pair=gate_pair,
            gate_cs=gate_cs,
            gate_ds=gate_ds,
            bias=bias,
        )


def _and(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None:
        return b
    if b is None:
        return a
    return a & b


def _add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


@dataclasses.dataclass(frozen=True, slots=True)
class Sinkhorn:
    """
    Soft-OT renormalisation wrapper for the differentiable path.

    Materialises the inner cost expression, runs ``n_iter`` Sinkhorn
    iterations in log space at temperature ``epsilon``, and returns the
    negative log transport plan as a new cost matrix. Useful as the cost
    feed into :class:`~unitrack.assignment.SoftAssignment`.

    Attributes
    ----------
    inner : ~unitrack.pipeline.CostProducer
        Wrapped cost producer.
    epsilon : float
        Entropy regularisation. Smaller is closer to a hard solution.
    n_iter : int
        Number of Sinkhorn iterations.

    """

    inner: typing.Any
    epsilon: float = 0.1
    n_iter: int = 50

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Run the inner producer and Sinkhorn-renormalise its cost.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to :attr:`inner` unchanged.

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix equal to the negative log transport
            plan. Inner gate masks are forwarded so downstream
            combinators retain provenance; ``bias`` is consumed by
            materialisation and intentionally dropped.

        """
        expr = self.inner(cs, ds, ctx)
        materialized = expr.materialize()
        log_plan = _sinkhorn_log_plan(materialized, self.epsilon, self.n_iter)
        # Forward the gate masks so downstream combinators retain the
        # "which rows/cols/pairs were blocked" provenance. ``bias`` is
        # consumed by ``materialize`` and intentionally dropped.
        return CostExpression.from_matrix(
            -log_plan,
            gate_pair=expr.gate_pair,
            gate_cs=expr.gate_cs,
            gate_ds=expr.gate_ds,
        )
