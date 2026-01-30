"""Implements memory that stores a Tensor value."""

import typing as T

import torch
from torch import Tensor, Size

from .base_state import DEFAULT_STATE_SLOTS, State

type DTypeSpec = torch.dtype | str


def _cast_dtype(dtype: DTypeSpec) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if not isinstance(dtype, torch.dtype):
        msg = f"No a valid data type: {dtype}!"
        raise ValueError(msg)
    return dtype


class Value(State):
    """A :class:`.State` that holds memory as-is without any updating logic
    implemented.
    """

    memory: Tensor
    index: Tensor

    def __init__(
        self,
        dtype: DTypeSpec = torch.float32,
        shape: Size | T.Iterable[int] = (),
        slots: int = DEFAULT_STATE_SLOTS,
    ):
        super().__init__()

        assert slots > 0, "Number of slots must be positive!"
        dtype = _cast_dtype(dtype)
        shape = (slots, *shape)
        alloc = torch.zeros(shape, dtype=dtype, requires_grad=False)
        self.register_buffer("memory", alloc, persistent=False)

    @property
    @T.override
    def slots(self) -> int:
        return self.memory.shape[0]

    @slots.setter
    @T.override
    def slots(self, slots: int) -> None:
        cur_slots = self.slots
        self.memory.resize_((slots, *self.memory.shape[1:]))
        if slots > cur_slots:
            self.memory[cur_slots:].zero_()

    @property
    @T.override
    def dtype(self) -> torch.dtype:
        return self.memory.dtype

    @property
    @T.override
    def shape(self) -> Size:
        return self.memory.shape[1:]

    @T.override
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.memory.dtype}, shape={self.memory.shape})"
        )

    @T.override
    def read(self, index: Tensor) -> dict[str, Tensor]:
        return {
            "memory": self.memory[index, ...],
        }

    @T.override
    def update(self, index: Tensor, memory: Tensor) -> None:
        assert index.shape[0] == memory.shape[0], (
            index.shape,
            memory.shape,
        )
        if self.memory.is_floating_point():
            memory = memory.type_as(self.memory)
        self._check_compatible(memory)
        self.memory[index] = memory

    @T.override
    def extend(self, index: Tensor, memory: Tensor) -> None:
        assert index.shape[0] == memory.shape[0], (
            index.shape,
            memory.shape,
        )
        if self.memory.is_floating_point():
            memory = memory.type_as(self.memory)
        self._check_compatible(memory)
        self.memory[index] = memory

    @T.override
    def observe(self, index: Tensor) -> Tensor:
        mem = self.memory[index, ...]
        assert mem.shape[1:] == self.shape, "Memory shape mismatch!"
        return mem

    @T.override
    def reset(self, index: Tensor) -> None:
        self.memory[index].zero_()
