from __future__ import annotations

import typing as T
from collections.abc import Sequence

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torch import nn

from .consts import KEY_ACTIVE, KEY_INDEX
from .debug import check_debug_enabled
from .stages import Stage

__all__ = ["MultiStageTracker", "SelectField"]


class SelectField(TensorDictModule):
    """Select fields from some input TensorDict and copy them to the output TensorDict."""

    def __init__(self, *keys: str, **keys_mapping: str):
        in_keys = []
        in_keys.extend(keys)
        in_keys.extend(keys_mapping.keys())

        out_keys = []
        out_keys.extend(keys)
        out_keys.extend(keys_mapping.values())

        super().__init__(lambda *args: args, in_keys=in_keys, out_keys=out_keys)


class MultiStageTracker(nn.Module):
    """Multi-stage tracker that applies a cascade of stages to a set of detections."""

    fields: nn.ModuleList  # [TensorDictModuleBase]
    stages: nn.ModuleList  # [Stage]

    def __init__(
        self, fields: Sequence[TensorDictModuleBase], stages: Sequence[Stage]
    ) -> None:
        super().__init__()

        assert len(stages) > 0

        self.fields = nn.ModuleList(fields)
        self.stages = nn.ModuleList(stages)

    @T.override
    def forward(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        inp: TensorDictBase,
        num: int,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """Perform tracking, returns a tuple of updated observations and the field-values
        of new tracklets.


        Parameters
        ----------
        ctx: TensorDictBase
            The current state's context
        obs: TensorDictBase
            The current state's observations (i.e. when ``.observe()`` is called on
            each field in memory)
        inp: TensorDictBase
            The state of the next frame, from which new detections are gathered at
            every field.
        num: int
            The amount of detections made. Fields must allocate within a TensorDict
            that enforces ``batch_size=[num]``.


        Returns
        -------
        Tuple[TensorDict, TensorDict]
            Updated observations and field-values of new tracklets.

        """
        if inp.device is None:
            msg = "Missing `device` attribute on inputs"
            raise ValueError(msg)

        # Create a dict of new tracklet candidates by passing the input state to
        # each field
        new_index = torch.arange(num, device=inp.device, dtype=torch.int64)
        new = TensorDict(
            {KEY_INDEX: new_index},
            batch_size=new_index.shape,
            device=inp.device,
        )

        obs = obs.to(device=inp.device)  # pyright: ignore[reportUnknownMemberType]

        for field in self.fields:
            field(inp, tensordict_out=new)

        assert obs.device is not None
        assert new.device is not None

        # Candidates for matching are all active observed tracklets
        active_mask = T.cast(torch.Tensor, obs.get(KEY_ACTIVE))
        obs_candidates = T.cast(TensorDictBase, obs._get_sub_tensordict(active_mask))
        for stage in self.stages:
            if obs_candidates.batch_size[0] == 0 or new.batch_size[0] == 0:
                break
            obs_candidates, new = stage(ctx, obs_candidates, new)

        obs_unmatched_mask = obs.get(KEY_INDEX) < 0

        if check_debug_enabled():
            pass
        obs[KEY_ACTIVE] = torch.where(obs_unmatched_mask, False, True)
        # obs.set_at_(KEY_ACTIVE, False, obs_unmatched_mask)
        # obs.set_at_(KEY_ACTIVE, True, ~obs_unmatched_mask)

        return obs, new
