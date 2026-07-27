"""
Stage-tree combinators.

Provides :class:`Pipe`, :class:`Sequential`, :class:`Parallel`,
:class:`Gated`, :class:`Filter`, and :class:`Iterate`.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing

import torch

from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    Gate,
    MatchOutcome,
    Tracklets,
)

from .base import Associator, CostProducer, GateProducer, PipelineTypeError

__all__ = ["Filter", "Gated", "Iterate", "Parallel", "Pipe", "Sequential"]

# Short aliases used in Gated dispatch helpers.
_GP = GateProducer
_CP = CostProducer
_Assoc = Associator
_MO = MatchOutcome
_PTE = PipelineTypeError


def _is_associator(obj: object) -> bool:
    """
    Return ``True`` iff *obj* satisfies the Associator contract.

    Runtime-checkable protocols only verify the presence of ``__call__``,
    not its signature, so a plain ``isinstance`` check cannot disambiguate
    :class:`~unitrack.pipeline.CostProducer` from
    :class:`~unitrack.pipeline.Associator`. The additional check here
    requires a ``cost`` keyword argument, which cost-producer leaves do
    not declare.
    """
    if not isinstance(obj, Associator):
        return False
    try:
        sig = inspect.signature(obj.__call__)
    except (ValueError, TypeError):
        return False
    return "cost" in sig.parameters


@dataclasses.dataclass(frozen=True, slots=True)
class Pipe:
    """
    Feed a :class:`~unitrack.pipeline.CostProducer` into an associator as one stage.

    The cost producer builds a :class:`~unitrack.data.CostExpression`,
    which is handed to the associator alongside the original
    ``(cs, ds, ctx)``.

    Attributes
    ----------
    cost : CostProducer
        Cost producer for this branch.
    assoc : Associator
        Associator that consumes the produced cost.

    """

    cost: CostProducer
    assoc: Associator

    def __post_init__(self) -> None:
        """
        Validate that ``cost`` and ``assoc`` satisfy their protocols.

        Raises
        ------
        PipelineTypeError
            If either field is not of the expected protocol type.

        """
        if not isinstance(self.cost, CostProducer):
            msg = f"Pipe.cost must be a CostProducer; got {type(self.cost).__name__}"
            raise PipelineTypeError(msg, path=["cost"])
        if not _is_associator(self.assoc):
            msg = f"Pipe.assoc must be an Associator; got {type(self.assoc).__name__}"
            raise PipelineTypeError(msg, path=["assoc"])

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
        cost: CostExpression | None = None,
    ) -> MatchOutcome:
        """
        Run the cost producer then the associator.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to both children.
        cost
            Ignored: a :class:`Pipe` owns its cost stage. The parameter
            exists so :class:`Pipe` satisfies the :class:`~unitrack.pipeline.Associator`
            signature.

        Returns
        -------
        MatchOutcome
            The associator's output.

        """
        del cost  # Pipe owns the cost stage; ignored if a parent sub-pipe passes one
        expr = self.cost(cs, ds, ctx)
        return self.assoc(cs, ds, ctx, expr)


def _classify_child(child: object) -> str:
    """
    Classify a Sequential child as ``"match"`` or ``"gate"``.

    A child accepting a ``cost`` keyword argument is treated as a match
    stage (Associator); otherwise it is a gate stage. The signature
    must be introspectable: a silent default would route an
    unclassifiable Associator through the gate-fold path and surface
    later as a confusing ``TypeError`` far from the root cause.

    Raises
    ------
    PipelineTypeError
        If the callable's signature is not introspectable.

    """
    try:
        sig = inspect.signature(child.__call__)  # type: ignore[union-attr]
    except (ValueError, TypeError) as e:
        msg = (
            f"Sequential child {type(child).__name__} has an un-introspectable "
            "__call__; cannot classify as match or gate. Wrap it in a regular "
            "Python function or set __signature__ explicitly."
        )
        raise PipelineTypeError(msg) from e
    return "match" if "cost" in sig.parameters else "gate"


class _SequentialBound:
    """
    Callable returned by ``Sequential[Gate]`` / ``Sequential[MatchOutcome]``.

    The subscript binds an expected child kind so the eventual
    ``Sequential(...)`` call validates children against it. Without this
    indirection, ``Sequential[Gate]([pipe])`` would silently run as a
    match cascade because both :class:`Pipe` and :class:`Sequential`
    accept a ``cost`` kwarg.
    """

    __slots__ = ("_cls", "_kind")

    def __init__(self, cls: type, kind: str) -> None:
        self._cls = cls
        self._kind = kind

    def __call__(self, children: list) -> Sequential:  # type: ignore[type-arg]
        return self._cls(children, expected_kind=self._kind)


class Sequential[T]:
    """
    Chain stages of the same output type.

    For ``T = MatchOutcome`` (the cascade), each child consumes the
    previous child's residuals. For ``T = Gate``, children are folded
    pointwise via :meth:`~unitrack.data.Gate.combine`.
    ``Sequential[CostExpression]`` is rejected; use :class:`Parallel`
    for cost-level merge.

    Parameters
    ----------
    children
        Stages of one output kind.
    expected_kind
        Internal flag bound by the subscript. ``None`` when
        :class:`Sequential` is constructed without a subscript; classified
        from the children in that case.

    Attributes
    ----------
    kind : str
        Either ``"match"`` or ``"gate"``.
    children : list
        Validated child stages.

    """

    def __class_getitem__(cls, item: type) -> typing.Any:
        """
        Bind the subscripted kind to instantiation.

        Raises
        ------
        PipelineTypeError
            If ``item`` is :class:`~unitrack.data.CostExpression`.

        """
        if item is CostExpression:
            msg = (
                "Sequential[CostExpression] is rejected; combine costs with "
                "Parallel instead"
            )
            raise PipelineTypeError(msg)
        if item is Gate:
            return _SequentialBound(cls, "gate")
        if item is MatchOutcome:
            return _SequentialBound(cls, "match")
        return super().__class_getitem__(item)  # type: ignore[misc]

    def __init__(self, children: list, *, expected_kind: str | None = None) -> None:
        """
        Validate that ``children`` share one output kind.

        Parameters
        ----------
        children
            Child stages; must be non-empty.
        expected_kind
            Optional pre-bound kind (``"match"`` or ``"gate"``) from the
            class subscript.

        Raises
        ------
        ValueError
            If ``children`` is empty.
        PipelineTypeError
            If children mix kinds or the inferred kind disagrees with
            ``expected_kind``.

        """
        if len(children) == 0:
            msg = "Sequential requires at least one child"
            raise ValueError(msg)
        kinds = [_classify_child(c) for c in children]
        if len(set(kinds)) != 1:
            msg = (
                f"Sequential children must all produce the same output type; "
                f"got {kinds!r}"
            )
            raise PipelineTypeError(msg)
        kind = kinds[0]
        if expected_kind is not None and kind != expected_kind:
            expected_param = {"gate": "Gate", "match": "MatchOutcome"}[expected_kind]
            got_param = {"gate": "Gate", "match": "MatchOutcome"}[kind]
            msg = (
                f"Sequential[{expected_param}] received {got_param}-typed children. "
                f"Children classified as {kinds!r}."
            )
            raise PipelineTypeError(msg)
        self.kind = kind
        self.children = list(children)

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
        cost: CostExpression | None = None,
    ) -> MatchOutcome | Gate:
        """
        Run every child in sequence.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to each child.
        cost
            Optional upstream cost. Passed only to the first cascade
            stage (subsequent stages operate on residuals with their
            own costs). Ignored on a gate cascade.

        Returns
        -------
        MatchOutcome or Gate
            On a match cascade, a single :class:`~unitrack.data.MatchOutcome` lifted to
            the original ``cs`` / ``ds`` row space. On a gate cascade,
            the pointwise combination of children's gates.

        """
        if self.kind == "match":
            # Forward the upstream ``cost`` (possibly carrying a gate applied
            # by a wrapping ``Gated``) to the first cascade stage. Subsequent
            # stages operate on residuals and build their own costs.
            return _run_sequential_match(self.children, cs, ds, ctx, cost=cost)
        if self.kind == "gate":
            del cost  # gate children don't consume an upstream cost
            results = [child(cs, ds, ctx) for child in self.children]
            folded = results[0]
            for r in results[1:]:
                folded = Gate.combine(folded, r)
            return folded
        msg = f"Sequential.kind={self.kind!r} not implemented"
        raise NotImplementedError(msg)


def _run_sequential_match(
    children: list,
    cs: Tracklets,
    ds: Detections,
    ctx: FrameContext,
    *,
    cost: CostExpression | None = None,
) -> MatchOutcome:
    """
    Run a cascaded chain of Associator children, threading residuals.

    Maintains residual-index mappings back to the original ``(cs, ds)``
    so callers see indices that reference the inputs they passed in,
    not the progressively-shrinking residuals inside the loop.

    The first child's ``soft_plan`` propagates through: it is the only
    plan in original ``(N, M)`` row space; later children's plans cover
    their residuals and cannot be losslessly scatter-merged back. A
    caller wanting a single full plan across a cascade should run a
    single :class:`~unitrack.assignment.SoftAssignment` instead.
    """
    n_orig = cs.batch_size[0]
    m_orig = ds.batch_size[0]
    device = cs.id.device

    cs_remap = torch.arange(n_orig, dtype=torch.int64, device=device)
    ds_remap = torch.arange(m_orig, dtype=torch.int64, device=device)
    matched_pairs = torch.zeros((0, 2), dtype=torch.int64, device=device)
    matched_costs: list[torch.Tensor] = []
    cost_dtype: torch.dtype | None = None
    first_soft_plan: torch.Tensor | None = None

    cur_cs = cs
    cur_ds = ds

    for idx, child in enumerate(children):
        if cur_cs.batch_size[0] == 0 or cur_ds.batch_size[0] == 0:
            break
        # Only the first child receives any upstream cost. After stage 1
        # consumes residuals, the upstream (full-size) cost no longer aligns
        # with cur_cs / cur_ds shapes, so subsequent stages must build their
        # own.
        outcome = (
            child(cur_cs, cur_ds, ctx, cost=cost)
            if idx == 0
            else child(cur_cs, cur_ds, ctx)
        )
        if idx == 0:
            first_soft_plan = outcome.soft_plan
        # Lift this child's matched pairs to the original space.
        if outcome.matched_pairs.shape[0] > 0:
            lifted = torch.stack(
                [
                    cs_remap[outcome.matched_pairs[:, 0]],
                    ds_remap[outcome.matched_pairs[:, 1]],
                ],
                dim=1,
            )
            matched_pairs = torch.cat([matched_pairs, lifted], dim=0)
            matched_costs.append(outcome.per_match_cost)
            if cost_dtype is None:
                cost_dtype = outcome.per_match_cost.dtype
        # Restrict cs/ds to this child's residuals; carry the remapping forward.
        cur_cs = cur_cs[outcome.tracklets_residual_index]
        cur_ds = cur_ds[outcome.detections_residual_index]
        cs_remap = cs_remap[outcome.tracklets_residual_index]
        ds_remap = ds_remap[outcome.detections_residual_index]

    if matched_costs:
        per_cost = torch.cat(matched_costs)
    else:
        per_cost = torch.zeros(0, dtype=cost_dtype or torch.float32, device=device)
    return MatchOutcome(
        matched_pairs=matched_pairs,
        tracklets_residual_index=cs_remap,
        detections_residual_index=ds_remap,
        per_match_cost=per_cost,
        soft_plan=first_soft_plan,
        batch_size=[],
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Filter:
    """
    Drop rows from ``cs`` and/or ``ds`` before a wrapped child runs.

    The wrapped child's :class:`~unitrack.data.MatchOutcome` indices are lifted back to
    the original (unfiltered) row space and the filtered-out rows are
    appended to the residual list, so the caller sees them as simply
    unmatched.

    Attributes
    ----------
    predicate : ~collections.abc.Callable or tuple of ~collections.abc.Callable
        Boolean predicate. For ``on='cs'`` the signature is
        ``predicate(cs) -> (N,) bool``; for ``on='ds'`` it is
        ``predicate(ds) -> (M,) bool``; for ``on='both'`` it is a tuple
        ``(p_cs, p_ds)``. ``True`` keeps the row.
    then : Associator
        Child stage to run on the filtered inputs.
    on : {'cs', 'ds', 'both'}
        Selects which side(s) to filter.

    """

    predicate: typing.Any
    then: typing.Any
    on: typing.Literal["cs", "ds", "both"] = "cs"

    def __post_init__(self) -> None:
        """
        Validate that ``then`` is an :class:`~unitrack.pipeline.Associator`.

        Raises
        ------
        PipelineTypeError
            If ``on`` is not one of ``{'cs', 'ds', 'both'}`` or
            ``then`` is not an associator.

        """
        if self.on not in ("cs", "ds", "both"):
            msg = f"Filter.on must be 'cs' | 'ds' | 'both'; got {self.on!r}"
            raise PipelineTypeError(msg, path=["on"])
        if not _is_associator(self.then):
            msg = (
                f"Filter.then must be an Associator (returns MatchOutcome); "
                f"got {type(self.then).__name__}"
            )
            raise PipelineTypeError(msg, path=["then"])

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
        cost: CostExpression | None = None,
    ) -> MatchOutcome:
        """
        Run the predicate, the child, and lift indices back.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to the child after filtering.
        cost
            Optional upstream cost; forwarded to the child unchanged
            when present.

        Returns
        -------
        MatchOutcome
            The child's outcome with indices remapped into the
            unfiltered row space and filtered-out rows appended as
            residuals.

        """
        keep_cs: torch.Tensor | None = None
        keep_ds: torch.Tensor | None = None

        if self.on == "cs":
            keep_cs = self.predicate(cs)
        elif self.on == "ds":
            keep_ds = self.predicate(ds)
        elif self.on == "both":
            p_cs, p_ds = self.predicate
            keep_cs = p_cs(cs)
            keep_ds = p_ds(ds)

        sub_cs = cs[keep_cs] if keep_cs is not None else cs
        sub_ds = ds[keep_ds] if keep_ds is not None else ds

        if cost is not None:
            outcome = self.then(sub_cs, sub_ds, ctx, cost)
        else:
            outcome = self.then(sub_cs, sub_ds, ctx)

        # Lift indices back to the unfiltered space, attaching the
        # filtered-out rows as residuals.
        if keep_cs is not None:
            cs_remap = torch.nonzero(keep_cs, as_tuple=False).squeeze(-1)
            unmatched_cs = torch.nonzero(~keep_cs, as_tuple=False).squeeze(-1)
            outcome = _remap_cs(outcome, cs_remap, unmatched_cs=unmatched_cs)
        if keep_ds is not None:
            ds_remap = torch.nonzero(keep_ds, as_tuple=False).squeeze(-1)
            unmatched_ds = torch.nonzero(~keep_ds, as_tuple=False).squeeze(-1)
            outcome = _remap_ds(outcome, ds_remap, unmatched_ds=unmatched_ds)
        return outcome


@dataclasses.dataclass(frozen=True, slots=True)
class Iterate:
    """
    Repeat a body stage ``n`` times against accumulating residuals.

    Attributes
    ----------
    n : int
        Number of iterations; must be positive.
    body : Associator
        Stage to invoke each iteration on the surviving residual.

    """

    n: int
    body: typing.Any

    def __post_init__(self) -> None:
        """
        Validate that ``body`` is an :class:`~unitrack.pipeline.Associator`.

        Raises
        ------
        ValueError
            If ``n`` is non-positive.
        PipelineTypeError
            If ``body`` is not an associator.

        """
        if self.n <= 0:
            msg = f"Iterate.n must be positive; got {self.n}"
            raise ValueError(msg)
        if not _is_associator(self.body):
            msg = f"Iterate.body must be an Associator; got {type(self.body).__name__}"
            raise PipelineTypeError(msg, path=["body"])

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
        cost: CostExpression | None = None,
    ) -> MatchOutcome:
        """
        Run :attr:`body` ``n`` times, cascading residuals across iterations.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to the body each iteration.
        cost
            Optional upstream cost (e.g. a gate-attached cost from a
            wrapping :class:`Gated`). Forwarded to the first iteration
            only; subsequent iterations operate on residuals and build
            their own cost expressions because the original full-size
            cost would shape-mismatch.

        Returns
        -------
        MatchOutcome
            Aggregated matched pairs and remaining residuals, lifted
            back to the original ``cs`` / ``ds`` row space.

        """
        device = cs.id.device
        n_orig = cs.batch_size[0]
        m_orig = ds.batch_size[0]

        cs_remap = torch.arange(n_orig, dtype=torch.int64, device=device)
        ds_remap = torch.arange(m_orig, dtype=torch.int64, device=device)
        matched_pairs = torch.zeros((0, 2), dtype=torch.int64, device=device)
        matched_costs: list[torch.Tensor] = []
        cost_dtype: torch.dtype | None = None
        first_soft_plan: torch.Tensor | None = None

        cur_cs = cs
        cur_ds = ds

        for idx in range(self.n):
            if cur_cs.batch_size[0] == 0 or cur_ds.batch_size[0] == 0:
                break
            outcome = (
                self.body(cur_cs, cur_ds, ctx, cost=cost)
                if idx == 0
                else self.body(cur_cs, cur_ds, ctx)
            )
            if idx == 0:
                first_soft_plan = outcome.soft_plan
            if outcome.matched_pairs.shape[0] > 0:
                lifted = torch.stack(
                    [
                        cs_remap[outcome.matched_pairs[:, 0]],
                        ds_remap[outcome.matched_pairs[:, 1]],
                    ],
                    dim=1,
                )
                matched_pairs = torch.cat([matched_pairs, lifted], dim=0)
                matched_costs.append(outcome.per_match_cost)
                if cost_dtype is None:
                    cost_dtype = outcome.per_match_cost.dtype
            cur_cs = cur_cs[outcome.tracklets_residual_index]
            cur_ds = cur_ds[outcome.detections_residual_index]
            cs_remap = cs_remap[outcome.tracklets_residual_index]
            ds_remap = ds_remap[outcome.detections_residual_index]

        per_cost = (
            torch.cat(matched_costs)
            if matched_costs
            else torch.zeros(0, dtype=cost_dtype or torch.float32, device=device)
        )
        return MatchOutcome(
            matched_pairs=matched_pairs,
            tracklets_residual_index=cs_remap,
            detections_residual_index=ds_remap,
            per_match_cost=per_cost,
            soft_plan=first_soft_plan,
            batch_size=[],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Gated:
    """
    Wrap a stage with a gate that projects the input or biases the cost.

    Per-side gates (``PerCs``, ``PerDs``) drop rows of ``cs`` / ``ds``
    before the child runs; pair and cost-bias gates attach to the
    downstream :class:`~unitrack.data.CostExpression`.

    Attributes
    ----------
    gate : GateProducer
        Gate producer evaluated on the full ``(cs, ds, ctx)``.
    then : Stage
        Body stage; may be a :class:`~unitrack.pipeline.CostProducer` or an
        :class:`~unitrack.pipeline.Associator`.

    """

    gate: typing.Any
    then: typing.Any

    def __post_init__(self) -> None:
        """
        Validate that :attr:`gate` is a :class:`GateProducer`.

        Raises
        ------
        PipelineTypeError
            If :attr:`gate` is not a gate producer.

        """
        if not isinstance(self.gate, _GP):
            msg = f"Gated.gate must be a GateProducer; got {type(self.gate).__name__}"
            raise _PTE(msg, path=["gate"])

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
        cost: CostExpression | None = None,
    ) -> MatchOutcome:
        """
        Evaluate the gate and dispatch on its variant.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to the gate and to :attr:`then`.
        cost
            Optional upstream cost. For per-pair / cost-bias gates with
            a non-Pipe body, the cost is gated and forwarded to the
            body.

        Returns
        -------
        MatchOutcome
            The body's outcome with indices lifted back to the original
            row space when a per-side gate dropped rows.

        """
        g = self.gate(cs, ds, ctx)
        return _gated_dispatch(g, self.then, cs=cs, ds=ds, ctx=ctx, cost=cost)


def _gated_dispatch(  # noqa: PLR0913
    g: typing.Any,
    body: typing.Any,
    *,
    cs: Tracklets,
    ds: Detections,
    ctx: FrameContext,
    cost: CostExpression | None,
) -> MatchOutcome | CostExpression:
    """
    Dispatch on the gate variant and run ``body`` accordingly.

    Two main paths apply, by ``body`` type. When ``body`` is a
    :class:`~unitrack.pipeline.CostProducer`, run it on the full ``(cs, ds)`` and attach
    the gate to the resulting :class:`~unitrack.data.CostExpression` (the gate's
    ``apply()`` handles all four variants uniformly). When ``body`` is
    an :class:`~unitrack.pipeline.Associator`, per-side gates filter ``cs`` / ``ds`` and
    remap indices back, while per-pair / cost-bias gates either
    intercept the inner cost producer of a :class:`Pipe` or fold the
    gate into an upstream ``cost`` for any other associator.
    """
    # Path 1: body produces a CostExpression — gate attaches to its output.
    if isinstance(body, _CP) and not _is_associator(body):
        expr = body(cs, ds, ctx)
        return g.apply(expr)

    # Path 2a: body is an Associator + per-side gate. Filter, run, remap.
    if g.kind == "per_cs":
        kept = g.mask
        new_cs = cs[kept]
        cs_remap = torch.nonzero(kept, as_tuple=False).squeeze(-1)
        if cost is not None:
            outcome = body(new_cs, ds, ctx, cost)
        else:
            outcome = body(new_cs, ds, ctx)
        return _remap_cs(
            outcome,
            cs_remap,
            unmatched_cs=torch.nonzero(~kept, as_tuple=False).squeeze(-1),
        )
    if g.kind == "per_ds":
        kept = g.mask
        new_ds = ds[kept]
        ds_remap = torch.nonzero(kept, as_tuple=False).squeeze(-1)
        if cost is not None:
            outcome = body(cs, new_ds, ctx, cost)
        else:
            outcome = body(cs, new_ds, ctx)
        return _remap_ds(
            outcome,
            ds_remap,
            unmatched_ds=torch.nonzero(~kept, as_tuple=False).squeeze(-1),
        )

    # Path 2b: body is an Associator + per_pair/cost_bias gate.
    if isinstance(body, Pipe):
        return _run_with_pair_or_bias(g, body, cs, ds, ctx)
    if _is_associator(body):
        if cost is not None:
            return body(cs, ds, ctx, g.apply(cost))
        msg = (
            "Gated.then must be a Pipe when gate kind is "
            "per_pair/cost_bias and no cost is supplied"
        )
        raise _PTE(msg, path=["then"])
    msg = f"Gated.then must be a Stage; got {type(body).__name__}"
    raise _PTE(msg, path=["then"])


def _run_with_pair_or_bias(
    g: typing.Any, body: Pipe, cs: Tracklets, ds: Detections, ctx: FrameContext
) -> MatchOutcome:
    """Apply a pair/bias gate to the cost expression produced inside a Pipe."""
    expr = body.cost(cs, ds, ctx)
    expr = g.apply(expr)
    return body.assoc(cs, ds, ctx, expr)


def _remap_cs(
    outcome: MatchOutcome,
    remap: typing.Any,
    *,
    unmatched_cs: typing.Any,
) -> MatchOutcome:
    """Remap cs indices in matched_pairs and residuals to original cs index space."""
    new_pairs = outcome.matched_pairs.clone()
    if new_pairs.shape[0] > 0:
        new_pairs[:, 0] = remap[new_pairs[:, 0]]
    return _MO(
        matched_pairs=new_pairs,
        tracklets_residual_index=torch.cat(
            [remap[outcome.tracklets_residual_index], unmatched_cs]
        ),
        detections_residual_index=outcome.detections_residual_index,
        per_match_cost=outcome.per_match_cost,
        soft_plan=outcome.soft_plan,
        batch_size=[],
    )


def _remap_ds(
    outcome: MatchOutcome,
    remap: typing.Any,
    *,
    unmatched_ds: typing.Any,
) -> MatchOutcome:
    """Remap ds indices in matched_pairs and residuals to original ds index space."""
    new_pairs = outcome.matched_pairs.clone()
    if new_pairs.shape[0] > 0:
        new_pairs[:, 1] = remap[new_pairs[:, 1]]
    return _MO(
        matched_pairs=new_pairs,
        tracklets_residual_index=outcome.tracklets_residual_index,
        detections_residual_index=torch.cat(
            [remap[outcome.detections_residual_index], unmatched_ds]
        ),
        per_match_cost=outcome.per_match_cost,
        soft_plan=outcome.soft_plan,
        batch_size=[],
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Parallel:
    """
    Cost-level merge of ``K`` branches into one :class:`~unitrack.data.CostExpression`.

    Attributes
    ----------
    children : list of CostProducer
        Branches; each emits a cost expression on the same ``(cs, ds)``.
    merge : Merge
        Reduction strategy applied to the branch outputs (e.g.
        :class:`~unitrack.pipeline.merge.WeightedSum`,
        :class:`~unitrack.pipeline.merge.Min`).

    """

    children: list
    merge: typing.Any

    def __post_init__(self) -> None:
        """
        Validate that every child is a :class:`~unitrack.pipeline.CostProducer`.

        Raises
        ------
        PipelineTypeError
            If any child is not a cost producer.

        """
        for i, c in enumerate(self.children):
            if not isinstance(c, _CP):
                name = type(c).__name__
                msg = f"Parallel.children[{i}] must be a CostProducer; got {name}"
                raise _PTE(msg, path=["children", str(i)])

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Run every child and reduce their cost expressions.

        Parameters
        ----------
        cs, ds, ctx
            Forwarded to every child.

        Returns
        -------
        CostExpression
            The merged ``(N, M)`` cost expression.

        """
        exprs = [c(cs, ds, ctx) for c in self.children]
        return self.merge(exprs)
