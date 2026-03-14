"""Typed snapshot of all tracked identities at a frame."""

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["Tracklets"]


_RESERVED_FIELDS: tuple[tuple[str, torch.dtype], ...] = (
    ("id", torch.int64),
    ("status", torch.int8),
    ("hits", torch.int32),
    ("time_since_update", torch.int32),
    ("age", torch.int32),
    ("frame_started", torch.int32),
    ("frame_last_seen", torch.int32),
)
_RESERVED_NAMES: frozenset[str] = frozenset(name for name, _ in _RESERVED_FIELDS)


def _validate_tracklets_schema(
    fields: dict[str, torch.Tensor], batch_size: list[int]
) -> None:
    """
    Validate ``fields`` against the Tracklets schema contract.

    Parameters
    ----------
    fields
        Mapping of field name to tensor value.
    batch_size
        The 1-D batch size declared at construction, ``[N]``.

    Raises
    ------
    ValueError
        If a reserved field is missing, has the wrong dtype, or has a
        leading dim that disagrees with ``batch_size[0]``.
    TypeError
        If a reserved field's value is not a :class:`torch.Tensor`.

    """
    if not batch_size or len(batch_size) == 0:
        msg = "Tracklets requires a 1-D batch_size like [N]; got " + repr(batch_size)
        raise ValueError(msg)
    n = batch_size[0]
    for name, dtype in _RESERVED_FIELDS:
        if name not in fields:
            msg = (
                f"Tracklets: missing reserved field {name!r}. "
                f"Reserved fields are {sorted(_RESERVED_NAMES)!r}; "
                "construct via Tracklets.empty() to get sensible zero defaults."
            )
            raise ValueError(msg)
        v = fields[name]
        if not isinstance(v, torch.Tensor):
            msg = (
                f"Tracklets: reserved field {name!r} must be a torch.Tensor; "
                f"got {type(v).__name__}"
            )
            raise TypeError(msg)
        if v.dtype != dtype:
            msg = (
                f"Tracklets: reserved field {name!r} must have dtype {dtype}; "
                f"got {v.dtype}"
            )
            raise ValueError(msg)
        if v.dim() < 1 or v.shape[0] != n:
            msg = (
                f"Tracklets: reserved field {name!r} must have leading dim {n} "
                f"(matching batch_size); got shape {tuple(v.shape)}"
            )
            raise ValueError(msg)


class Tracklets(TensorDict):
    """
    Snapshot of all live tracklets at one frame.

    Reserved fields are common to every Tracker; user fields are added by
    Tracker construction (one per declared :class:`~unitrack.states.State`) and live
    alongside the reserved set in a flat namespace.

    Construction via ``Tracklets(...)`` validates the reserved-field
    schema: all reserved names must be present with the documented dtype
    and a leading dim matching ``batch_size[0]``. :meth:`empty` is the
    canonical way to build a fresh snapshot without re-typing every
    reserved field.

    Attributes
    ----------
    id : torch.Tensor
        Tracklet IDs, ``int64`` shape ``(N,)``.
    status : torch.Tensor
        Lifecycle status as :class:`~unitrack.lifecycle.TrackletStatus`
        integer codes, ``int8`` shape ``(N,)``.
    hits : torch.Tensor
        Cumulative match count per tracklet, ``int32`` shape ``(N,)``.
    time_since_update : torch.Tensor
        Frames elapsed since the last match, ``int32`` shape ``(N,)``.
    age : torch.Tensor
        Total age in frames, ``int32`` shape ``(N,)``.
    frame_started : torch.Tensor
        Frame index at which the tracklet was created, ``int32`` shape
        ``(N,)``.
    frame_last_seen : torch.Tensor
        Frame index at which the tracklet was last matched, ``int32``
        shape ``(N,)``.

    Notes
    -----
    The class subclasses :class:`~tensordict.TensorDict` rather than
    using ``@tensorclass`` so the per-tracker user-field set can be
    declared dynamically. Schema validation runs on user-facing
    construction; tensordict-internal paths (slicing, ``torch.cat``,
    ``to``) reuse already-validated rows and skip the re-check.

    """

    def __init__(self, **kwargs: object) -> None:
        # `batch_size` is the tell-tale kwarg in user-facing construction;
        # TensorDict-internal slicing / cat / to bypass __init__ entirely.
        if "batch_size" in kwargs:
            tensor_fields = {
                k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
            }
            _validate_tracklets_schema(tensor_fields, list(kwargs["batch_size"]))  # type: ignore[arg-type]
        super().__init__(**kwargs)  # type: ignore[arg-type]

    # Reserved field property accessors — documented dtypes are contracts.
    @property
    def id(self) -> torch.Tensor:
        """``int64`` shape ``(N,)`` tracklet IDs."""
        return self["id"]  # type: ignore[invalid-return-type]

    @property
    def status(self) -> torch.Tensor:
        """``int8`` shape ``(N,)`` lifecycle status codes."""
        return self["status"]  # type: ignore[invalid-return-type]

    @property
    def hits(self) -> torch.Tensor:
        """``int32`` shape ``(N,)`` cumulative match counts."""
        return self["hits"]  # type: ignore[invalid-return-type]

    @property
    def time_since_update(self) -> torch.Tensor:
        """``int32`` shape ``(N,)`` frames since the last match."""
        return self["time_since_update"]  # type: ignore[invalid-return-type]

    @property
    def age(self) -> torch.Tensor:
        """``int32`` shape ``(N,)`` total ages in frames."""
        return self["age"]  # type: ignore[invalid-return-type]

    @property
    def frame_started(self) -> torch.Tensor:
        """``int32`` shape ``(N,)`` frame indices at creation."""
        return self["frame_started"]  # type: ignore[invalid-return-type]

    @property
    def frame_last_seen(self) -> torch.Tensor:
        """``int32`` shape ``(N,)`` frame indices of the last match."""
        return self["frame_last_seen"]  # type: ignore[invalid-return-type]

    def __getattr__(self, name: str) -> torch.Tensor:
        """Look up *name* as a user field via :class:`TensorDict` indexing."""
        # Avoid infinite recursion during object init and for dunder/private attrs.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]  # type: ignore[invalid-return-type]
        except KeyError:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg) from None

    @classmethod
    def empty(  # type: ignore[invalid-method-override]
        cls,
        *,
        device: torch.types.Device | None = None,
        user_fields: dict[str, torch.Tensor] | None = None,
    ) -> Tracklets:
        """
        Construct a zero-row Tracklets snapshot.

        Parameters
        ----------
        device
            Device for the zero-row reserved tensors. ``None`` selects
            the default device.
        user_fields
            Mapping from user-field name to a zero-row tensor template.
            Each value supplies the dtype and trailing shape for that
            field.

        Returns
        -------
        Tracklets
            A :class:`~unitrack.data.Tracklets` with ``batch_size=[0]``, all reserved
            fields zero-initialised, and the supplied user fields
            attached.

        """

        def z(dt: torch.dtype) -> torch.Tensor:
            return torch.zeros(0, dtype=dt, device=device)

        kwargs: dict[str, torch.Tensor] = {
            name: z(dtype) for name, dtype in _RESERVED_FIELDS
        }
        if user_fields:
            kwargs.update(user_fields)
        return cls(**kwargs, batch_size=[0])  # type: ignore[invalid-argument-type]
