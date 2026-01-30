from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, override

import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor, nn

if TYPE_CHECKING:
    from . import MultiStageTracker, TrackletMemory

__all__ = ["SimpleTracker", "StatefulTracker"]


def _maybe_read_item[T: int | float | bool](
    t: type[T], *args: Tensor | T
) -> Iterator[T]:
    for a in args:
        if isinstance(a, Tensor):
            yield t(a.item())
        else:
            assert isinstance(a, t), type(a)
            yield a


def _assert_result_count(ids: Tensor, expected: int):
    if ids.shape[0] != expected:
        msg = f"Expected {expected} IDs to be returned, but got {len(ids)} instead!"
        raise ValueError(msg)


##################
# Simple wrapper #
##################


class SimpleTracker(nn.Module):
    """
    A wrapper around tracker and memory.

    Uses only one memory for all tracking bookkeeping.

    Practically, this means that it only supports inference settings where each sequence
    of frames in processed in isolation, e.g. first all frames of sequence 1, then all
    frames of sequence 2, etc.

    This should be preferred over :class:`StatefulTracker` where applicable, because
    this removes the need to manage multiple memory instances of the tracklet state
    buffers.
    """

    tracker: MultiStageTracker
    memory: TrackletMemory
    device: torch.types.Device | None
    last_key: int | None

    def __init__(
        self,
        tracker: MultiStageTracker,
        memory: TrackletMemory,
        *,
        device: torch.types.Device | None = "cpu",
    ) -> None:
        """
        Initialize the SimpleTracker.

        Parameters
        ----------
        tracker
            The tracker to be used.
        memory
            The memory to be used.
        device
            The device to be used for the tracker and memory. By default, all buffers
            and parameters (if any) are moved to the CPU.

        """
        super().__init__()

        self.tracker = tracker
        self.memory = memory
        self.device = torch.device(device) if device is not None else None
        self.last_key = None

    @override
    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)

        if self.device is not None:
            self.memory = self.memory.to(self.device)
            self.tracker = self.tracker.to(self.device)

    @torch.compiler.disable()  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    def _reset_on_new(self, key: int) -> bool:
        if self.last_key != key:
            self.memory.reset()
            self.last_key = key

            return True
        return False

    @override
    def forward(self, x: TensorDict, n: int, key: int, frame: int) -> TensorDict:
        """
        Perform tracking step.

        Parameters
        ----------
        x
            Represents the state of the current iteration.
        n
            The amount of detections in the current frame, amounts to the length of
            IDs returned
        key
            The key that identifies the sequence in which the detections have been made
        frame
            The current frame number, used to identify the temporal position within
            the sequence.

        Returns
        -------
        Tensor[n]
            Assigned instance IDs

        """
        key, frame = _maybe_read_item(int, key, frame)
        _ = self._reset_on_new(key)

        state_ctx, state_obs = self.memory.read(frame)
        state_obs, new = self.tracker(state_ctx, state_obs, x, n)

        return self.memory.write_with_observe(state_ctx, state_obs, new)

    if TYPE_CHECKING:
        __call__ = forward  # pyright: ignore[reportUnannotatedClassAttribute]


####################
# Stateful tracker #
####################


class _MemoryReadWriter(nn.Module):
    """
    Wrapper around a tracklet memory.

    Enables writing and reading in the forward pass.

    This is necessary because the memory is a stateful module that is not compatible
    with the functional API.
    """

    def __init__(self, memory: TrackletMemory):
        super().__init__()
        self.memory = memory

    @override
    def forward(
        self,
        write: bool,
        transaction: int | tuple[TensorDictBase, TensorDictBase, TensorDictBase],
    ):
        if write:
            ctx, obs, new = transaction
            return self.memory.write_with_observe(ctx, obs, new)

        (frame,) = transaction
        return self.memory.read(frame)


class StatefulTracker(nn.Module):
    """
    A wrapper around tracker and tracklets.

    Enables stateful tracking of objects for every separate sequence.
    """

    mem_buffers: Mapping[str | int, dict[str, torch.Tensor]]
    mem_params: dict[str, torch.Tensor]

    def __init__(self, tracker: MultiStageTracker, memory: TrackletMemory):
        super().__init__()

        self.tracker = tracker
        self.memory_delegate = _MemoryReadWriter(memory)
        self.memory_storage: (
            tuple[dict[str, torch.nn.Parameter], dict[str, Tensor], dict[str, Tensor]]
            | None
        ) = None

    @classmethod
    def _split_persistent_buffers(
        cls, module: nn.Module, prefix: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Split the buffers of a module into two dictionaries.

        One for buffers that are shared across sequences (persistent) and one for
        buffers that are unique to every sequence (non-persistent).
        """
        shared = {}
        unique = {}

        for name, buf in module.named_buffers(prefix="", recurse=False):
            name_prefixed = f"{prefix}.{name}"
            if name in module._non_persistent_buffers_set:
                unique[name_prefixed] = buf
            else:
                shared[name_prefixed] = buf

        for name, submodule in module.named_children():
            s, u = cls._split_persistent_buffers(submodule, prefix=f"{prefix}.{name}")
            shared.update(s)
            unique.update(u)

        return shared, unique

    def _lookup_pbd(
        self, key: int
    ) -> tuple[
        dict[str, torch.nn.Parameter], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]:
        if self.memory_storage is None:
            prefix = "memory"
            memory = self.memory_delegate.memory
            params = dict(memory.named_parameters(prefix=prefix))
            buffers_shared, buffers_unique = self._split_persistent_buffers(
                memory, prefix=prefix
            )
            buffers_unique_default = copy.deepcopy(buffers_unique)
            buffers_unique_map = defaultdict(
                lambda: copy.deepcopy(buffers_unique_default)
            )

            self.memory_storage = (params, buffers_shared, buffers_unique_map)
        else:
            params, buffers_shared, buffers_unique_map = self.memory_storage

        return params, buffers_shared, buffers_unique_map[key]

    @override
    def forward(self, x: TensorDict, n: int, key: int, frame: int) -> Tensor:
        """
        Perform tracking step.

        Parameters
        ----------
        x
            Represents the state of the current iteration.
        n
            The amount of detections in the current frame, amounts to the length of
            IDs returned
        key
            The key that identifies the sequence in which the detections have been made
        frame
            The current frame number, used to identify the temporal position within
            the sequence.

        Returns
        -------
        Tensor[n]
            Assigned instance IDs

        """
        key, frame = _maybe_read_item(int, key, frame)
        params, buffers_shared, buffers_unique = self._lookup_pbd(key)

        # Create parameter-buffer-dict (PBD) for functionally calling the tracker
        # modules, which uses the buffers that are isolated for each sequence
        pbd: dict[str, torch.Tensor] = {**params, **buffers_shared, **buffers_unique}

        # Run the tracker using the functional API
        state_ctx, state_obs = torch.func.functional_call(
            self.memory_delegate, pbd, (False, (frame,)), strict=True
        )
        state_obs, new = self.tracker(state_ctx, state_obs, x, n)
        ids: torch.Tensor = torch.func.functional_call(
            self.memory_delegate, pbd, (True, (state_ctx, state_obs, new)), strict=True
        )

        # Write back the updated state to the current sequence's buffers
        for buf_key, buf_val in pbd.items():
            if buf_key in params:
                continue
            if buf_key in buffers_shared:
                buffers_shared[buf_key] = buf_val
                continue
            if buf_key in buffers_unique:
                buffers_unique[buf_key] = buf_val
                continue

            msg = (
                f"Buffer with key {buf_key!r} not found in parameters and buffers dict!"
            )
            raise KeyError(msg)

        _assert_result_count(ids, n)
        return ids
