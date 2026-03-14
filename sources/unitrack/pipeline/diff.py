"""Hard-to-soft node-replacement registry for ``differentiable=True``."""

from __future__ import annotations

import dataclasses
import typing

from unitrack.assignment import Associate, SoftAssignment
from unitrack.gates import MotionGate
from unitrack.gates.soft import SoftMotionGate
from unitrack.lifecycle import StandardLifecycle
from unitrack.lifecycle.soft import SoftLifecycle
from unitrack.states import Replace, SoftReplace, State

__all__ = [
    "SoftRegistry",
    "default_soft_registry",
    "validate_soft_tree",
    "walk_swap",
    "walk_swap_states",
]

_Factory = typing.Callable[[object], object]


@dataclasses.dataclass(slots=True)
class SoftRegistry:
    """
    Registry mapping hard node types to soft-replacement factories.

    Attributes
    ----------
    table : dict
        Mapping from hard-node class to a factory ``(hard_node) -> soft_node``.

    """

    table: dict[type, _Factory] = dataclasses.field(
        default_factory=dict,
    )

    def register(self, hard_cls: type, factory: _Factory) -> None:
        """
        Register a factory for the given hard-node class.

        Parameters
        ----------
        hard_cls
            Class to swap on encounter.
        factory
            Callable that consumes the hard node and returns its soft
            replacement.

        """
        self.table[hard_cls] = factory

    def soft_for(self, node: object) -> object:
        """
        Return the soft replacement for ``node``, or ``node`` unchanged.

        Parameters
        ----------
        node
            Candidate node.

        Returns
        -------
        object
            Replacement produced by the matching factory, or ``node``
            itself if no factory applies.

        """
        for cls, factory in self.table.items():
            if isinstance(node, cls):
                return factory(node)
        return node  # passthrough — not all nodes need swapping


def default_soft_registry(
    *,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    sinkhorn_iters: int = 50,
) -> SoftRegistry:
    r"""
    Build the default hard-to-soft registry for differentiable mode.

    Parameters
    ----------
    epsilon
        Entropy regularisation for the Sinkhorn-backed
        :class:`~unitrack.assignment.SoftAssignment` swap of
        :class:`~unitrack.assignment.Associate`. Smaller is sharper
        (closer to a hard assignment); larger is smoother.
    temperature
        Temperature for the
        :class:`~unitrack.gates.soft.SoftMotionGate` swap of
        :class:`~unitrack.gates.MotionGate`. The Mahalanobis
        :math:`\\chi^2` is divided by this before attachment as a soft
        cost bias.
    sinkhorn_iters
        Number of Sinkhorn iterations for the soft assignment.

    Returns
    -------
    SoftRegistry
        A registry that swaps :class:`~unitrack.assignment.Associate` →
        :class:`~unitrack.assignment.SoftAssignment`,
        :class:`~unitrack.gates.MotionGate` →
        :class:`~unitrack.gates.SoftMotionGate`,
        :class:`~unitrack.states.Replace` → :class:`~unitrack.states.SoftReplace`,
        and :class:`~unitrack.lifecycle.StandardLifecycle` →
        :class:`~unitrack.lifecycle.SoftLifecycle`.

    """
    reg = SoftRegistry()

    def _associate_to_soft(node: Associate) -> Associate:
        # Preserve the user's threshold on the hard Associate so a tree built
        # with `Associate(Hungarian(threshold=0.7))` doesn't silently lose
        # 0.7 when switched into differentiable mode.
        return Associate(
            SoftAssignment(
                threshold=float(node.assignment.threshold),
                epsilon=epsilon,
                num_iter=sinkhorn_iters,
            )
        )

    reg.register(Associate, _associate_to_soft)

    def _motion_to_soft(node: MotionGate) -> SoftMotionGate:
        return SoftMotionGate(
            mean_field=node.mean_field,
            cov_field=node.cov_field,
            temperature=temperature,
        )

    reg.register(MotionGate, _motion_to_soft)

    def _replace_to_soft(node: Replace) -> SoftReplace:
        return SoftReplace(field=node.field)

    reg.register(Replace, _replace_to_soft)

    def _lifecycle_to_soft(node: StandardLifecycle) -> SoftLifecycle:
        return SoftLifecycle(
            min_hits=node.min_hits,
            max_age=node.max_age,
            grace_period=node.grace_period,
            allow_reid=node.allow_reid,
        )

    reg.register(StandardLifecycle, _lifecycle_to_soft)

    return reg


def _recurse(root: object, registry: SoftRegistry) -> object:  # noqa: PLR0911
    """Apply structural recursion to a node that was NOT itself swapped."""
    if hasattr(root, "children"):
        new_children = [walk_swap(c, registry) for c in root.children]
        if dataclasses.is_dataclass(root):
            return dataclasses.replace(root, children=new_children)  # type: ignore[arg-type]
        return type(root)(new_children)
    if hasattr(root, "cost") and hasattr(root, "assoc"):
        return dataclasses.replace(
            root,
            cost=walk_swap(root.cost, registry),  # type: ignore[union-attr]
            assoc=walk_swap(root.assoc, registry),  # type: ignore[union-attr]
        )
    if hasattr(root, "gate") and hasattr(root, "then"):
        return dataclasses.replace(
            root,
            gate=walk_swap(root.gate, registry),  # type: ignore[union-attr]
            then=walk_swap(root.then, registry),  # type: ignore[union-attr]
        )
    if hasattr(root, "body"):
        return dataclasses.replace(
            root,
            body=walk_swap(root.body, registry),  # type: ignore[union-attr]
        )
    if hasattr(root, "predicate") and hasattr(root, "then"):
        return dataclasses.replace(
            root,
            then=walk_swap(root.then, registry),  # type: ignore[union-attr]
        )
    return root


def walk_swap(root: object, registry: SoftRegistry) -> object:
    """
    Replace nodes in the stage tree by structural recursion.

    Uses :func:`dataclasses.replace` to preserve frozen-dataclass
    immutability.

    Parameters
    ----------
    root
        Stage-tree root.
    registry
        Hard-to-soft factory table.

    Returns
    -------
    object
        New root with all replaceable nodes swapped.

    """
    swapped = registry.soft_for(root)
    if swapped is not root:
        return swapped
    return _recurse(root, registry)


def walk_swap_states(
    states: dict[str, State],
    registry: SoftRegistry,
) -> dict[str, State]:
    """
    Swap a :class:`~unitrack.states.State`'s ``process``, ``observation``, ``init``.

    Parameters
    ----------
    states
        Mapping of state name to :class:`~unitrack.states.State`.
    registry
        Hard-to-soft factory table.

    Returns
    -------
    dict
        New mapping with each :class:`~unitrack.states.State` rebuilt via
        :func:`dataclasses.replace`; the input is not mutated.

    """
    out: dict[str, State] = {}
    for name, st in states.items():
        out[name] = dataclasses.replace(
            st,
            process=walk_swap(st.process, registry),  # type: ignore[arg-type]
            observation=walk_swap(st.observation, registry),  # type: ignore[arg-type]
            init=walk_swap(st.init, registry),  # type: ignore[arg-type]
        )
    return out


_HARD_FORBIDDEN: tuple[type, ...] = (MotionGate, Replace, StandardLifecycle)


def _collect_hard_residuals(node: object, path: str, out: list[str]) -> None:
    """Walk a stage subtree and record any hard nodes that the swap missed."""
    if isinstance(node, _HARD_FORBIDDEN):
        out.append(f"{path}: {type(node).__name__}")
    # Associate-wrapped hard Assignments (e.g. a user's custom Hungarian
    # subclass) are also forbidden — the registry already swaps Associate
    # itself, but a residual one indicates the structural recursion didn't
    # reach this branch.
    if isinstance(node, Associate) and not isinstance(node.assignment, SoftAssignment):
        out.append(
            f"{path}.assignment: {type(node.assignment).__name__} "
            "(expected SoftAssignment)"
        )
    if hasattr(node, "children"):
        for i, c in enumerate(node.children):  # type: ignore[attr-defined]
            _collect_hard_residuals(c, f"{path}.children[{i}]", out)
    for attr in ("cost", "assoc", "gate", "then", "body", "predicate", "inner"):
        if hasattr(node, attr):
            _collect_hard_residuals(getattr(node, attr), f"{path}.{attr}", out)


def validate_soft_tree(
    root: object, states: dict[str, State], lifecycle: object
) -> None:
    """
    Raise if the differentiable tree still contains a hard component.

    Run after :func:`walk_swap` / :func:`walk_swap_states` to catch
    user-defined containers the structural recursion missed, or hard
    Assignment subclasses tucked behind a non-default Associate. The
    check keeps differentiable training honest: a caller seeing a NaN
    loss can rule out a silently-retained hard node.

    Parameters
    ----------
    root
        Stage-tree root after the swap.
    states
        State mapping after the swap.
    lifecycle
        Lifecycle node after the swap.

    Raises
    ------
    TypeError
        If any forbidden hard node survives.

    """
    residuals: list[str] = []
    _collect_hard_residuals(root, "root", residuals)
    _collect_hard_residuals(lifecycle, "lifecycle", residuals)
    for name, st in states.items():
        _collect_hard_residuals(st.process, f"states[{name}].process", residuals)
        _collect_hard_residuals(
            st.observation, f"states[{name}].observation", residuals
        )
        _collect_hard_residuals(st.init, f"states[{name}].init", residuals)
    if residuals:
        bullets = "\n  - ".join(residuals)
        msg = (
            "differentiable=True: hard nodes survived walk_swap:\n  - "
            f"{bullets}\nRegister soft replacements in the SoftRegistry or "
            "swap them in your tree before constructing the Tracker."
        )
        raise TypeError(msg)
