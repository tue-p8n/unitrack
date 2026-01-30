r"""
Implements the `Lost` stage.

Filters out candidates that have been lost for more than a configurable maximum amount
of time.
"""

from __future__ import annotations

import typing as T  # noqa: N812

from tensordict import TensorDictBase

from ..consts import KEY_DELTA, KEY_FRAME
from .base_stage import Stage

__all__ = ["Lost"]


class Lost(Stage):
    """
    Stage that filters out candidates.

    Filters out candidates that have been lost for less than a configurable maximum
    amount.

    This is useful to remove candidates that have been lost for too long without
    removing them directly in the tracklet memory itself, which also allows for a
    maximum retention time before a candidate is removed.


    The time lost is computed via $$ T_l = t - T_c - dt $$ where $t$ is the
    current time (i.e. frame) and $T_c$ are the candidate time states.
    The value is corrected with time-step $dt$ to account for candidates not yet
    being updated to the current time, as the state update is performed
    *after* all stages have been ran.
    """

    max_lost: T.Final[int]

    def __init__(self, max_lost: int):
        """
        Initialize the Lost stage.

        Parameters
        ----------
        max_lost
            Maximum amount of time a candidate may remain lost.

        """
        super().__init__()

        assert max_lost > 0, max_lost

        self.max_lost = max_lost

    def forward(
        self, ctx: TensorDictBase, cs: TensorDictBase, ds: TensorDictBase
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """Filter out candidates that have been lost for too long."""
        if len(cs) == 0:
            return cs, ds

        time_lost = ctx.get(KEY_FRAME) - cs.get(KEY_FRAME) - ctx.get(KEY_DELTA)
        # Debug prints removed to clean up, assuming logical fix or I will re-add
        # if needed
        return cs[time_lost <= self.max_lost], ds