"""Typed tensor shape and dtype declaration used by State schemas."""

from __future__ import annotations

import typing

import torch

__all__ = ["TensorSpec"]


class TensorSpec(typing.NamedTuple):
    """
    Per-tracklet shape and dtype declaration.

    The slot dimension (number of tracklets) is implicit; :attr:`shape`
    describes the trailing dimensions of one tracklet's value of this
    field.

    Attributes
    ----------
    shape : tuple of int
        Trailing dimensions of one tracklet's value.
    dtype : torch.dtype
        Element dtype.

    """

    shape: tuple[int, ...]
    dtype: torch.dtype

    def empty(
        self, slots: int, device: torch.types.Device | None = None
    ) -> torch.Tensor:
        """
        Allocate a zero buffer matching this spec.

        Parameters
        ----------
        slots
            Leading slot dimension (number of tracklets).
        device
            Device for the allocation.

        Returns
        -------
        torch.Tensor
            Zero tensor of shape ``(slots, *self.shape)`` and dtype
            :attr:`dtype`.

        """
        return torch.zeros((slots, *self.shape), dtype=self.dtype, device=device)
