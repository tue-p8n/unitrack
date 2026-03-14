"""Identity ``Process``, ``Replace``, and simple ``Initializer`` recipes."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import (
    Detections,
    FrameContext,
    MatchOutcome,
    TensorSpec,
    Tracklets,
)

__all__ = [
    "ConstantInitializer",
    "EyeInitializer",
    "FromDetectionField",
    "Identity",
    "NoopObservation",
    "NoopProcess",
    "NormalizedFromDetectionField",
    "PadZerosInitializer",
    "Replace",
    "ScaledFromDetectionField",
    "ZerosInitializer",
]


@dataclasses.dataclass(frozen=True, slots=True)
class Identity:
    """
    No-op :class:`~unitrack.states.Process` for fields that need no prediction step.

    Useful for embeddings, one-hot class labels, scores, and any other
    field whose value does not evolve under the motion model.

    Parameters
    ----------
    field : str
        Tracklet field name (unused; kept for symmetry with the matching
        observation).

    """

    field: str

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Return tracklets unchanged."""
        del ctx
        return cs


@dataclasses.dataclass(frozen=True, slots=True)
class Replace:
    """
    Hard-replace :class:`Observation`: matched tracklets adopt the detection value.

    Parameters
    ----------
    field : str
        Tracklet field to overwrite from the matched detection's field of
        the same name.

    """

    field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Replace matched tracklet field values with detection values."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        new_field = getattr(cs, self.field).clone()
        new_field[cs_idx] = getattr(ds, self.field)[ds_idx]
        return cs.set(self.field, new_field)


@dataclasses.dataclass(frozen=True, slots=True)
class FromDetectionField:
    """
    :class:`Initializer` that copies a named field from new detections.

    Parameters
    ----------
    field : str
        Detection field whose tensor seeds the new tracklets.

    """

    field: str

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return the detection field tensor."""
        del ctx
        return getattr(ds, self.field)


@dataclasses.dataclass(frozen=True, slots=True)
class ZerosInitializer:
    """
    :class:`Initializer` that fills the schema-shaped buffer with zeros.

    Parameters
    ----------
    schema : ~unitrack.data.TensorSpec
        Per-tracklet shape and dtype.

    """

    schema: TensorSpec

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return a zeros buffer shaped for the given detections batch."""
        del ctx
        return self.schema.empty(slots=ds.batch_size[0], device=ds.index.device)


@dataclasses.dataclass(frozen=True, slots=True)
class EyeInitializer:
    """
    :class:`Initializer` that emits a per-tracklet scaled identity matrix.

    Used to initialise Kalman covariances. Each new tracklet receives
    ``scale * I_dim``.

    Parameters
    ----------
    dim : int
        Side length of the identity matrix.
    scale : float, optional
        Multiplier applied to ``I_dim``. Default ``1.0``.

    """

    dim: int
    scale: float = 1.0

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return ``(N, dim, dim)`` per-tracklet scaled identity matrices."""
        del ctx
        n = ds.batch_size[0]
        eye = torch.eye(self.dim, device=ds.index.device) * self.scale
        return eye.unsqueeze(0).expand(n, self.dim, self.dim).contiguous()


@dataclasses.dataclass(frozen=True, slots=True)
class PadZerosInitializer:
    """
    :class:`Initializer` that copies a detection field and zero-pads to ``full_dim``.

    Used to spawn a Kalman state from a measurement-only detection: the
    detection's measurement vector becomes the leading slice of the full
    state, with zeros for the unobserved velocity (or higher-order)
    components. ``ds.{field}`` must have shape ``(N, meas_dim)`` and
    ``meas_dim <= full_dim``.

    Parameters
    ----------
    field : str
        Detection field that supplies the leading measurement slice.
    full_dim : int
        Final state dimensionality.

    Raises
    ------
    ValueError
        If the detection field is wider than ``full_dim``.

    """

    field: str
    full_dim: int

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return a ``(N, full_dim)`` tensor with detection data padded with zeros."""
        del ctx
        meas = getattr(ds, self.field)
        n, d_meas = meas.shape
        if d_meas > self.full_dim:
            msg = (
                f"PadZerosInitializer({self.field!r}, full_dim={self.full_dim}): "
                f"detection field has {d_meas} dims > full_dim"
            )
            raise ValueError(msg)
        if d_meas == self.full_dim:
            return meas.clone()
        pad = torch.zeros(
            n, self.full_dim - d_meas, dtype=meas.dtype, device=meas.device
        )
        return torch.cat([meas, pad], dim=-1)


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantInitializer:
    """
    :class:`Initializer` that fills the schema-shaped buffer with a constant.

    Used to seed scalar auxiliary fields such as a von Mises-Fisher
    concentration or a gallery fill-count. The fill value is cast to the
    schema dtype.

    Parameters
    ----------
    schema : ~unitrack.data.TensorSpec
        Per-tracklet shape and dtype.
    value : float
        Constant fill value.

    """

    schema: TensorSpec
    value: float

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return a constant-filled buffer shaped for the given detections batch."""
        del ctx
        buf = self.schema.empty(slots=ds.batch_size[0], device=ds.index.device)
        return buf.fill_(self.value)


@dataclasses.dataclass(frozen=True, slots=True)
class NormalizedFromDetectionField:
    """
    :class:`Initializer` copying a detection field, L2-normalised along the last axis.

    Seeds a unit-vector field (e.g. a von Mises-Fisher mean direction) from
    a possibly unnormalised detection embedding.

    Parameters
    ----------
    field : str
        Detection field whose tensor seeds the new tracklets.
    eps : float, optional
        Lower bound on the norm used for normalisation. Default ``1e-12``.

    """

    field: str
    eps: float = 1e-12

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return the L2-normalised detection field tensor."""
        del ctx
        return torch.nn.functional.normalize(
            getattr(ds, self.field), dim=-1, eps=self.eps
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ScaledFromDetectionField:
    """
    :class:`Initializer` copying a detection field scaled by a constant.

    Used to seed an information-vector field ``y = z / init_var`` from a
    measurement ``z``.

    Parameters
    ----------
    field : str
        Detection field whose tensor seeds the new tracklets.
    scale : float
        Multiplier applied to the field.

    """

    field: str
    scale: float

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return the scaled detection field tensor."""
        del ctx
        return getattr(ds, self.field) * self.scale


@dataclasses.dataclass(frozen=True, slots=True)
class NoopProcess:
    """
    :class:`~unitrack.states.Process` that leaves the snapshot unchanged.

    Useful for the auxiliary covariance entry of a Kalman state, where
    the paired mean entry's :class:`KalmanLinear` predict already writes
    the new covariance — running another predict here would double-step.
    """

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        """Return the snapshot unchanged."""
        del ctx
        return cs


@dataclasses.dataclass(frozen=True, slots=True)
class NoopObservation:
    """
    :class:`Observation` that leaves the snapshot unchanged.

    The dual of :class:`NoopProcess` for the update step.
    """

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Return the snapshot unchanged."""
        del ds, match, ctx
        return cs
