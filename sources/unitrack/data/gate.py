"""Algebraic ``Gate`` variant: ``PerPair``, ``PerCs``, ``PerDs``, ``CostBias``."""

from __future__ import annotations

import dataclasses
import typing

import torch

from .cost import CostExpression

__all__ = ["Gate"]


@dataclasses.dataclass(frozen=True, slots=True)
class _PerPair:
    """
    Per-pair feasibility gate.

    Attributes
    ----------
    mask : torch.Tensor
        Bool ``(N, M)`` mask. ``True`` is allowed.
    kind : str
        Discriminator literal ``"per_pair"``.

    """

    mask: torch.Tensor
    kind: typing.Literal["per_pair"] = "per_pair"

    def apply(self, expr: CostExpression) -> CostExpression:
        """Attach the mask to ``expr.gate_pair`` (AND-combined if already set)."""
        new_pair = self.mask if expr.gate_pair is None else expr.gate_pair & self.mask
        return CostExpression.from_matrix(
            expr.matrix,
            gate_pair=new_pair,
            gate_cs=expr.gate_cs,
            gate_ds=expr.gate_ds,
            bias=expr.bias,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _PerCs:
    """
    Per-tracklet feasibility gate.

    Attributes
    ----------
    mask : torch.Tensor
        Bool ``(N,)`` mask. ``True`` keeps the tracklet row.
    kind : str
        Discriminator literal ``"per_cs"``.

    """

    mask: torch.Tensor
    kind: typing.Literal["per_cs"] = "per_cs"

    def apply(self, expr: CostExpression) -> CostExpression:
        """Attach the mask to ``expr.gate_cs`` (AND-combined if already set)."""
        new_cs = self.mask if expr.gate_cs is None else expr.gate_cs & self.mask
        return CostExpression.from_matrix(
            expr.matrix,
            gate_pair=expr.gate_pair,
            gate_cs=new_cs,
            gate_ds=expr.gate_ds,
            bias=expr.bias,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _PerDs:
    """
    Per-detection feasibility gate.

    Attributes
    ----------
    mask : torch.Tensor
        Bool ``(M,)`` mask. ``True`` keeps the detection column.
    kind : str
        Discriminator literal ``"per_ds"``.

    """

    mask: torch.Tensor
    kind: typing.Literal["per_ds"] = "per_ds"

    def apply(self, expr: CostExpression) -> CostExpression:
        """Attach the mask to ``expr.gate_ds`` (AND-combined if already set)."""
        new_ds = self.mask if expr.gate_ds is None else expr.gate_ds & self.mask
        return CostExpression.from_matrix(
            expr.matrix,
            gate_pair=expr.gate_pair,
            gate_cs=expr.gate_cs,
            gate_ds=new_ds,
            bias=expr.bias,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _CostBias:
    """
    Additive cost-bias gate.

    Attributes
    ----------
    matrix : torch.Tensor
        Float ``(N, M)`` bias to add at materialisation.
    kind : str
        Discriminator literal ``"cost_bias"``.

    """

    matrix: torch.Tensor
    kind: typing.Literal["cost_bias"] = "cost_bias"

    def apply(self, expr: CostExpression) -> CostExpression:
        """Add ``matrix`` to ``expr.bias`` (sum-combined if already set)."""
        new_bias = self.matrix if expr.bias is None else expr.bias + self.matrix
        return CostExpression.from_matrix(
            expr.matrix,
            gate_pair=expr.gate_pair,
            gate_cs=expr.gate_cs,
            gate_ds=expr.gate_ds,
            bias=new_bias,
        )


_GateAny = typing.Union["_PerPair", "_PerCs", "_PerDs", "_CostBias", "_PairAndBias"]


class Gate:
    """
    Algebraic gate closed under conjunction via :meth:`~unitrack.data.Gate.combine`.

    The four constructors are exposed as nested classes so the public
    spelling is ``Gate.PerPair(...)``, ``Gate.CostBias(...)``, and so on.
    """

    PerPair = _PerPair
    PerCs = _PerCs
    PerDs = _PerDs
    CostBias = _CostBias

    @staticmethod
    def combine(a: _GateAny, b: _GateAny) -> _GateAny:
        """
        Combine two gate operands via mask AND and bias sum.

        Parameters
        ----------
        a, b
            Any two values drawn from ``{PerPair, PerCs, PerDs, CostBias,
            _PairAndBias}``.

        Returns
        -------
        ~unitrack.data.Gate
            Another value of the same family. Mask-only operands stay
            mask-only with the most specific kind preserved (e.g. two
            ``PerCs`` combine to a ``PerCs``); mixing a mask with a bias
            yields a ``_PairAndBias``.

        """
        if a.kind == "cost_bias" and b.kind == "cost_bias":
            return _CostBias(matrix=a.matrix + b.matrix)

        # Pure-mask paths preserve the most specific kind (no premature
        # promotion to PerPair when both are PerCs, etc).
        masks_only = {a.kind, b.kind} <= {"per_pair", "per_cs", "per_ds"}
        if masks_only:
            return _combine_masks(a, b)  # type: ignore[arg-type]

        # Mixed mask + bias (incl. _PairAndBias): split into mask part and
        # bias part, combine each independently, then re-pack.
        a_mask = _extract_pair_mask(a)
        b_mask = _extract_pair_mask(b)
        a_bias = _extract_bias(a)
        b_bias = _extract_bias(b)

        mask = _and_or_promote(a_mask, b_mask)
        bias = _add_or_promote(a_bias, b_bias)

        if mask is not None and bias is not None:
            # Broadcast mask to bias shape so apply() can ``&`` it directly.
            return _PairAndBias(
                mask=mask.expand_as(bias).contiguous()
                if mask.shape != bias.shape
                else mask,
                bias_matrix=bias,
            )
        if bias is not None:
            return _CostBias(matrix=bias)
        if mask is None:
            msg = (
                "Gate.combine: internal error — mask is None in mask-only fallback. "
                "This branch should be unreachable; please file a bug."
            )
            raise RuntimeError(msg)
        return _PerPair(mask=mask)


def _and_or_promote(
    acc: torch.Tensor | None, new: torch.Tensor | None
) -> torch.Tensor | None:
    """Return ``acc & new`` when both present; otherwise the one that exists."""
    if acc is None:
        return new
    if new is None:
        return acc
    return acc & new


def _add_or_promote(
    acc: torch.Tensor | None, new: torch.Tensor | None
) -> torch.Tensor | None:
    """Return ``acc + new`` when both present; otherwise the one that exists."""
    if acc is None:
        return new
    if new is None:
        return acc
    return acc + new


def _extract_pair_mask(g: _GateAny | _PairAndBias) -> torch.Tensor | None:
    """Extract a broadcastable (N, M) mask from any gate, or None."""
    if g.kind == "per_pair":
        return g.mask  # type: ignore[union-attr]
    if g.kind == "per_cs":
        return g.mask[:, None]  # type: ignore[union-attr]
    if g.kind == "per_ds":
        return g.mask[None, :]  # type: ignore[union-attr]
    if g.kind == "pair_and_bias":
        return g.mask  # type: ignore[union-attr]
    return None  # cost_bias


def _extract_bias(g: _GateAny | _PairAndBias) -> torch.Tensor | None:
    """Extract an additive (N, M) bias from any gate, or None."""
    if g.kind == "cost_bias":
        return g.matrix  # type: ignore[union-attr]
    if g.kind == "pair_and_bias":
        return g.bias_matrix  # type: ignore[union-attr]
    return None


def _combine_masks(a: _GateAny, b: _GateAny) -> _GateAny:
    if a.kind == "per_cs" and b.kind == "per_cs":
        return _PerCs(mask=a.mask & b.mask)
    if a.kind == "per_ds" and b.kind == "per_ds":
        return _PerDs(mask=a.mask & b.mask)
    if a.kind == "per_pair" and b.kind == "per_pair":
        return _PerPair(mask=a.mask & b.mask)
    # Cross-kind promotions:
    pair = _to_pair_mask(a)
    other = _to_pair_mask(b)
    if pair is None or other is None:
        msg = f"unsupported cross-kind combine: {a.kind!r}, {b.kind!r}"
        raise NotImplementedError(msg)
    return _PerPair(mask=pair & other)


def _to_pair_mask(g: _GateAny) -> torch.Tensor | None:
    """Promote a single-kind gate to (N, M) bool mask via outer-AND broadcast."""
    if g.kind == "per_pair":
        return g.mask  # type: ignore[union-attr]
    if g.kind == "per_cs":
        return g.mask[:, None]  # type: ignore[union-attr]
    if g.kind == "per_ds":
        return g.mask[None, :]  # type: ignore[union-attr]
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class _PairAndBias:
    """
    Joint mask-and-bias gate produced by mixed-kind combination.

    Returned by :meth:`Gate.combine` when one operand is a ``CostBias``
    and the other is any mask kind. Both pieces are applied atomically.

    Attributes
    ----------
    mask : torch.Tensor
        Bool ``(N, M)`` feasibility mask (broadcast to pair shape if the
        source mask was per-side).
    bias_matrix : torch.Tensor
        Float ``(N, M)`` additive bias.
    kind : str
        Discriminator literal ``"pair_and_bias"``.

    """

    mask: torch.Tensor
    bias_matrix: torch.Tensor
    kind: typing.Literal["pair_and_bias"] = "pair_and_bias"

    def apply(self, expr: CostExpression) -> CostExpression:
        """Attach both the pair mask and the bias atomically."""
        new_pair = self.mask if expr.gate_pair is None else expr.gate_pair & self.mask
        new_bias = (
            self.bias_matrix if expr.bias is None else expr.bias + self.bias_matrix
        )
        return CostExpression.from_matrix(
            expr.matrix,
            gate_pair=new_pair,
            gate_cs=expr.gate_cs,
            gate_ds=expr.gate_ds,
            bias=new_bias,
        )
