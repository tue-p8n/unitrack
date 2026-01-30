from __future__ import annotations

import typing as T
from abc import abstractmethod
from collections.abc import Iterable

import torch
import typing_extensions as TX
from tensordict import TensorDictBase

from unitrack.consts import KEY_INDEX
from unitrack.debug import check_debug_enabled

__all__ = ["Stage"]


class Stage(torch.nn.Module):
    """Base class for stages in a ::class::`..Tracker`.

    Inputs to a stage are in the context data, or as fields of the detections.
    """

    def __init__(
        self,
        required_fields: Iterable[str] = (),
    ):
        super().__init__()

        self.required_fields = list(required_fields)

    @TX.override
    def extra_repr(self) -> str:
        req = ", ".join(self.required_fields)
        return f"fields=[{req}]"

    @abstractmethod
    @TX.override
    def forward(
        self,
        ctx: TensorDictBase,
        cs: TensorDictBase,
        ds: TensorDictBase,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        raise NotImplementedError

    def match(self, cs: TensorDictBase, ds: TensorDictBase) -> None:
        """Match candidates to detections. Propagates data and IDs from detections to candidates.

        Parameters
        ----------
        cs
            Candidates
        ds
            Detections

        """
        if check_debug_enabled():
            pass

        cs_keys = T.cast(list[str], cs.keys())
        ds_keys = T.cast(list[str], ds.keys())
        for key in ds_keys:
            if key.startswith("_"):
                continue
            if key not in cs_keys:
                continue
            cs.set_(key, ds.get(key))

        # pass the index of the detection such that it can be retrieved later when
        # collecting the retuned ID
        indices = ds.get(KEY_INDEX)
        cs.set_(KEY_INDEX, ds.get(KEY_INDEX))
        ds.set_(KEY_INDEX, torch.full_like(indices, -1))
