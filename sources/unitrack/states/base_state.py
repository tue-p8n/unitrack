"""Base state classes."""

from __future__ import annotations

import typing as T  # noqa: N812
from abc import abstractmethod

import torch
from torch import Tensor

__all__ = ["DEFAULT_STATE_SLOTS", "State", "StateValue"]


DEFAULT_STATE_SLOTS: T.Final[int] = 1024 * 4


type StateValue = Tensor | dict[str, Tensor]


class State(torch.nn.Module):
    """Abstract base class for state modules."""

    def __init__(self, **kwargs):
        """
        Initialize the State module.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the parent class.

        """
        super().__init__(**kwargs)

    def _check_compatible(self, memory: Tensor, *, raises: bool = True) -> bool:
        """
        Check if the given memory is compatible with the state.

        Parameters
        ----------
        memory
            The memory to check compatibility with.
        raises
            Whether to raise an exception if the memory is incompatible.

        Returns
        -------
            Whether the memory is compatible.

        Raises
        ------
        ValueError
            If the memory tensor is incompatible with the state.
            Only raised if :param:`raises` is True.

        """
        if memory.dtype != self.dtype:
            if raises:
                msg = (
                    f"Memory data type {memory.dtype} is incompatible with state"
                    f" data type {self.dtype}"
                )
                raise ValueError(msg)
            return False
        if memory.shape[1:] != self.shape:
            if raises:
                msg = (
                    f"Memory shape {memory.shape[1:]} is incompatible with state"
                    f" shape {self.shape}"
                )
                raise ValueError(msg)
            return False
        return True

    @property
    @abstractmethod
    def slots(self) -> int:
        """The number of slots in the state."""
        raise NotImplementedError

    @slots.setter
    @abstractmethod
    def slots(self, slots: int) -> None:
        """Set the number of slots in the state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> T.Iterable[int]:
        """The shape of the state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The data type of the state."""
        raise NotImplementedError

    @abstractmethod
    def update(self, index: Tensor, update: StateValue) -> None:
        """Update the current state from the given update tensor."""
        raise NotImplementedError

    @abstractmethod
    def extend(self, index: Tensor, extend: StateValue) -> None:
        """Extend the current state from the given extend tensor."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, index: Tensor) -> StateValue:
        """
        Observe the current state.

        Returns
        -------
            State value

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, index: Tensor):
        """
        Reset the state to its initial value.

        Essentially sets the buffers to their initial value.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, index: Tensor) -> dict[str, Tensor]:
        """
        Read the current state.

        Returns
        -------
            State value

        """
        raise NotImplementedError

    def evolve(self, index: Tensor, delta: Tensor):  # noqa: ARG002
        """
        Evolve the state by the given delta. This is a no-op by default.

        Parameters
        ----------
        index
            Indices of the state to evolve.
        delta
            The time difference to evolve the state by.

        """
        return self.observe(index)