"""
Gallery (feature-bank) state — keep a ring buffer of recent embeddings.

The non-parametric memory-bank approach to appearance: instead of a single
filtered embedding, each tracklet stores its last ``K`` matched embeddings.
Matching then consults the whole buffer (see
:class:`~unitrack.costs.GalleryCost`), so one good past view re-associates an
object whose current appearance has drifted. The primary field still holds
the most recent embedding, so plain :class:`~unitrack.costs.Cosine` matching
also works as a fallback.
"""

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

from .base import State
from .identity import (
    ConstantInitializer,
    FromDetectionField,
    NoopObservation,
    NoopProcess,
)

__all__ = [
    "GalleryAppend",
    "GalleryInitializer",
    "gallery_state_entries",
]


@dataclasses.dataclass(frozen=True, slots=True)
class GalleryAppend:
    """
    Update step: push each matched detection's embedding into the ring buffer.

    For each matched pair the detection embedding overwrites the oldest
    slot (``count mod K``), the fill count is incremented, and the primary
    field is set to the new embedding (the most-recent view). Unmatched
    tracklets keep their gallery.

    Parameters
    ----------
    field : str
        Primary tracklet field (most recent embedding). The gallery lives
        in ``f"{field}_gallery"`` and the count in ``f"{field}_count"``.
    meas_field : str
        Detection field holding the embedding to append.

    """

    field: str
    meas_field: str

    @property
    def gallery_field(self) -> str:
        """Return the gallery-buffer field name."""
        return f"{self.field}_gallery"

    @property
    def count_field(self) -> str:
        """Return the fill-count field name."""
        return f"{self.field}_count"

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Append matched detection embeddings to their tracklet galleries."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        z = getattr(ds, self.meas_field)[ds_idx]  # (Kp, D)
        gallery = getattr(cs, self.gallery_field).clone()  # (N, K, D)
        count = getattr(cs, self.count_field).clone()  # (N,)
        emb = getattr(cs, self.field).clone()  # (N, D)

        capacity = gallery.shape[1]
        slot = count[cs_idx] % capacity  # (Kp,)
        gallery[cs_idx, slot] = z
        count[cs_idx] = count[cs_idx] + 1
        emb[cs_idx] = z
        return (
            cs.set(self.field, emb)
            .set(self.gallery_field, gallery)
            .set(self.count_field, count)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GalleryInitializer:
    """
    :class:`~unitrack.states.Initializer` seeding a gallery with one embedding.

    The new tracklet's gallery has its first slot set to the detection
    embedding and the rest zeroed; the paired count initialiser starts at 1.

    Parameters
    ----------
    field : str
        Detection field supplying the seed embedding.
    capacity : int
        Number of gallery slots ``K``.
    dim : int
        Embedding dimensionality ``D``.

    """

    field: str
    capacity: int
    dim: int

    def __call__(self, ds: Detections, ctx: FrameContext) -> torch.Tensor:
        """Return a ``(N, K, D)`` gallery with slot 0 seeded from the detection."""
        del ctx
        z = getattr(ds, self.field)  # (N, D)
        n = z.shape[0]
        buf = torch.zeros(n, self.capacity, self.dim, dtype=z.dtype, device=z.device)
        if n > 0:
            buf[:, 0, :] = z
        return buf


def gallery_state_entries(
    field: str,
    *,
    dim: int,
    capacity: int = 8,
    meas_field: str | None = None,
) -> dict[str, State]:
    """
    Build the ``(recent, gallery, count)`` state entries for a feature bank.

    Pair the gallery with :class:`~unitrack.costs.GalleryCost` (reading
    ``f"{field}_gallery"`` and ``f"{field}_count"``) for memory-bank
    matching, or match the primary ``field`` with plain
    :class:`~unitrack.costs.Cosine` for last-embedding matching.

    Parameters
    ----------
    field : str
        Name of the primary (most-recent) field and prefix for the
        ``f"{field}_gallery"`` / ``f"{field}_count"`` auxiliaries.
    dim : int
        Embedding dimensionality.
    capacity : int
        Number of gallery slots ``K``.
    meas_field : str, optional
        Detection field supplying the embedding. Defaults to
        :paramref:`field`.

    Returns
    -------
    dict
        Three :class:`~unitrack.states.State` entries keyed by ``field``,
        ``f"{field}_gallery"`` and ``f"{field}_count"``.

    """
    meas_field = meas_field or field
    return {
        field: State(
            schema=TensorSpec(shape=(dim,), dtype=torch.float32),
            process=NoopProcess(),
            observation=GalleryAppend(field=field, meas_field=meas_field),
            init=FromDetectionField(meas_field),
        ),
        f"{field}_gallery": State(
            schema=TensorSpec(shape=(capacity, dim), dtype=torch.float32),
            process=NoopProcess(),
            observation=NoopObservation(),
            init=GalleryInitializer(field=meas_field, capacity=capacity, dim=dim),
        ),
        f"{field}_count": State(
            schema=TensorSpec(shape=(), dtype=torch.int64),
            process=NoopProcess(),
            observation=NoopObservation(),
            init=ConstantInitializer(TensorSpec(shape=(), dtype=torch.int64), 1),
        ),
    }
