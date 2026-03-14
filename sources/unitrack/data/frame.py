"""Per-frame timing and stream metadata."""

from __future__ import annotations

import torch
from tensordict import tensorclass

__all__ = ["FrameContext"]


@tensorclass
class FrameContext:
    """
    Per-frame context threaded through the entire stage tree.

    Carries timing (frame index, delta, fps) and stream identity. Stages
    that need frame-level information (e.g. time-aware covariance
    scaling, annealed thresholds) read it from the third argument of
    their ``__call__``.

    Attributes
    ----------
    frame_idx : torch.Tensor
        Scalar ``int64`` frame index.
    delta : torch.Tensor
        Scalar ``float32`` seconds since the previous frame.
    fps : torch.Tensor
        Scalar ``float32`` frame rate.
    stream_key : torch.Tensor
        Scalar ``int64`` stream identifier.

    """

    frame_idx: torch.Tensor
    delta: torch.Tensor
    fps: torch.Tensor
    stream_key: torch.Tensor

    @classmethod
    def make(
        cls,
        frame_idx: int,
        *,
        delta: float | None = None,
        fps: float = 1.0,
        stream_key: int = 0,
        device: torch.types.Device | None = None,
    ) -> FrameContext:
        """
        Build a :class:`FrameContext` from Python scalars.

        Parameters
        ----------
        frame_idx
            Frame index.
        delta
            Seconds since the previous frame. Defaults to ``1.0 / fps``,
            the natural step for a real frame stream. Pass ``0.0`` to
            freeze a Kalman predict step explicitly.
        fps
            Frame rate. Must be positive when ``delta`` is omitted.
        stream_key
            Stream identifier; relevant for multi-stream wrappers.
        device
            Device for the packed scalar tensors.

        Returns
        -------
        FrameContext
            The packed scalar context.

        Raises
        ------
        ValueError
            If ``delta`` is omitted and ``fps`` is non-positive.

        """
        if delta is None:
            if fps <= 0:
                msg = (
                    f"FrameContext.make: fps must be positive when delta is not "
                    f"given explicitly; got fps={fps}"
                )
                raise ValueError(msg)
            delta = 1.0 / fps
        return cls(
            frame_idx=torch.tensor(frame_idx, dtype=torch.int64, device=device),
            delta=torch.tensor(delta, dtype=torch.float32, device=device),
            fps=torch.tensor(fps, dtype=torch.float32, device=device),
            stream_key=torch.tensor(stream_key, dtype=torch.int64, device=device),
            batch_size=[],  # type: ignore[unknown-argument]
        )
