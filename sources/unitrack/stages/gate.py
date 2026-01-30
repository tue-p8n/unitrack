from __future__ import annotations

import abc
import typing as T

import torch
import typing_extensions as TX
from tensordict import TensorDictBase
from torch import nn

from ..consts import KEY_INDEX
from .base_stage import Stage


class GateModule(nn.Module):
    """A module that performs gating of observations, returns a mask for candidates and
    detections.
    """

    @abc.abstractmethod
    @TX.override
    def forward(
        self, ctx: TensorDictBase, cs: TensorDictBase, ds: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class Gate(Stage):
    """A stage that performs gating of observations."""

    def __init__(self, gate: nn.Module, then: T.Sequence[Stage]):
        # TODO: required fields?
        super().__init__()
        self.gate = gate
        self.then = nn.ModuleList(then)

    @TX.override
    def extra_repr(self) -> str:
        repr = super().extra_repr()
        return repr + f"gate={self.gate}\nthen={self.then}"

    @TX.override
    def forward(
        self,
        ctx: TensorDictBase,
        cs: TensorDictBase,
        ds: TensorDictBase,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        # Perform gating
        cs_mask, ds_mask = self.gate(ctx, cs, ds)
        cs_gated = cs._get_sub_tensordict(cs_mask)
        ds_gated = ds._get_sub_tensordict(ds_mask)

        # Perform subsequent stages on only the subset
        for stage in self.then:
            if cs_gated.batch_size[0] == 0 or ds_gated.batch_size[0] == 0:
                break
            cs_gated, ds_gated = stage(ctx, cs_gated, ds_gated)

        # Continue to next stage, but remove all observations that have been assigned
        # already
        cs_index = cs.get(KEY_INDEX)
        cs_unmatched = cs._get_sub_tensordict(cs_index < 0)

        ds_index = ds.get(KEY_INDEX)
        ds_unmatched = ds._get_sub_tensordict(ds_index >= 0)

        return cs_unmatched, ds_unmatched
