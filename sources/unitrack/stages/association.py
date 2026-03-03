"""Association stage."""

from __future__ import annotations

import typing as T  # noqa: N812

import typing_extensions as TX  # noqa: N812
from tensordict import TensorDictBase

if T.TYPE_CHECKING:
    from unitrack.assignment import Assignment
    from unitrack.costs import Cost

from unitrack.debug import check_debug_enabled

from .base_stage import Stage

__all__ = ["Association"]


class Association(Stage):
    """
    An association stage.

    Assigns new track IDs using a cost matrix computed via a :class:`.Cost` module.
    """

    def __init__(
        self,
        cost: Cost,
        assignment: Assignment,
    ) -> None:
        """
        Initialize the Association stage.

        Parameters
        ----------
        cost
            Cost module.
        assignment
            Assignment module.

        """
        super().__init__()

        self.cost = cost
        self.assignment = assignment

        self._extend_required_fields(self.cost)

    def _extend_required_fields(self, *costs: Cost) -> None:
        for cost in costs:
            self.required_fields += cost.required_fields

    @TX.override
    def forward(
        self, ctx: TensorDictBase, cs: TensorDictBase, ds: TensorDictBase
    ) -> tuple[TensorDictBase, TensorDictBase]:
        if check_debug_enabled():
            pass

        if len(cs) == 0 or len(ds) == 0:
            return cs, ds

        cost_matrix = self.cost(cs, ds)

        matches, cs_fail_idx, ds_fail_idx = self.assignment(cost_matrix)

        if len(matches) > 0:
            cs_match = cs._get_sub_tensordict(matches[:, 0])
            ds_match = ds._get_sub_tensordict(matches[:, 1])

            self.match(cs_match, ds_match)

        cs_fail = cs._get_sub_tensordict(cs_fail_idx)
        ds_fail = ds._get_sub_tensordict(ds_fail_idx)

        return cs_fail, ds_fail
