"""Typed record of one frame's new detections."""

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["Detections"]


def _validate_detections_schema(
    fields: dict[str, torch.Tensor], batch_size: list[int]
) -> None:
    """
    Validate ``fields`` against the Detections schema contract.

    Parameters
    ----------
    fields
        Mapping of field name to tensor value.
    batch_size
        The 1-D batch size declared at construction, ``[M]``.

    Raises
    ------
    ValueError
        If the reserved ``index`` field is missing, has the wrong dtype,
        or has a leading dim that disagrees with ``batch_size[0]``.
    TypeError
        If ``index`` is not a :class:`torch.Tensor`.

    """
    if not batch_size or len(batch_size) == 0:
        msg = "Detections requires a 1-D batch_size like [M]; got " + repr(batch_size)
        raise ValueError(msg)
    m = batch_size[0]
    if "index" not in fields:
        msg = (
            "Detections: missing reserved field 'index' (int64 (M,)). "
            "Construct via Detections.empty() to get a sensible zero default."
        )
        raise ValueError(msg)
    v = fields["index"]
    if not isinstance(v, torch.Tensor):
        msg = f"Detections: 'index' must be a torch.Tensor; got {type(v).__name__}"
        raise TypeError(msg)
    if v.dtype != torch.int64:
        msg = f"Detections: 'index' must have dtype torch.int64; got {v.dtype}"
        raise ValueError(msg)
    if v.dim() < 1 or v.shape[0] != m:
        msg = (
            f"Detections: 'index' must have leading dim {m} "
            f"(matching batch_size); got shape {tuple(v.shape)}"
        )
        raise ValueError(msg)


class Detections(TensorDict):
    """
    Record of one frame's detections.

    User fields match the :class:`~unitrack.data.Tracklets` schema of the same Tracker.
    The ``index`` field carries the caller's per-detection ordering and
    is threaded through to :class:`~unitrack.data.MatchOutcome` so the caller can
    recover it after assignment.

    Construction via ``Detections(...)`` validates the reserved-field
    schema: ``index`` must be present, ``int64``, and shape ``(M,)``.

    Attributes
    ----------
    index : torch.Tensor
        ``int64`` shape ``(M,)`` caller-assigned detection index.

    """

    def __init__(self, **kwargs: object) -> None:
        # `batch_size` is the tell-tale kwarg in user-facing construction;
        # TensorDict-internal slicing / cat / to bypass __init__ entirely.
        if "batch_size" in kwargs:
            tensor_fields = {
                k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
            }
            _validate_detections_schema(tensor_fields, list(kwargs["batch_size"]))  # type: ignore[arg-type]
        super().__init__(**kwargs)  # type: ignore[arg-type]

    @property
    def index(self) -> torch.Tensor:
        """``int64`` shape ``(M,)`` caller-assigned detection index."""
        return self["index"]  # type: ignore[invalid-return-type]

    def __getattr__(self, name: str) -> torch.Tensor:
        """Look up *name* as a user field via :class:`TensorDict` indexing."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]  # type: ignore[invalid-return-type]
        except KeyError:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg) from None

    @classmethod
    def empty(  # type: ignore[invalid-method-override]
        cls, *, device: torch.types.Device | None = None
    ) -> Detections:
        """
        Construct a zero-row Detections record.

        Parameters
        ----------
        device
            Device for the zero-row ``index`` tensor. ``None`` selects
            the default device.

        Returns
        -------
        Detections
            A :class:`~unitrack.data.Detections` with ``batch_size=[0]`` and an empty
            ``index`` field.

        """
        return cls(
            index=torch.zeros(0, dtype=torch.int64, device=device),
            batch_size=[0],
        )
