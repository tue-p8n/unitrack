"""
Gallery (feature-bank) cost — match a detection against a tracklet's history.

Single-embedding matching (one stored vector per tracklet) is brittle when
an object's appearance changes: a fresh detection from a new viewpoint can
look unlike the last stored embedding even though it is the same identity.
DeepSORT / MeMOT-style trackers keep a small **gallery** of recent
embeddings per tracklet and match a detection against the *best* (or mean)
of them, so a single good historical view is enough to re-associate.

:class:`GalleryCost` reads a tracklet's gallery buffer and fill count
(maintained by :class:`~unitrack.states.GalleryAppend`) and reduces the
per-slot cosine similarities to an ``(N, M)`` cost matrix.
"""

from __future__ import annotations

import dataclasses
import typing

import torch

from unitrack.data import CostExpression, Detections, FrameContext, Tracklets

__all__ = ["GalleryCost"]

GalleryReduce = typing.Literal["max", "mean"]


@dataclasses.dataclass(frozen=True, slots=True)
class GalleryCost:
    """
    Cosine cost between detections and each tracklet's embedding gallery.

    For tracklet gallery ``(N, K, D)`` and detections ``(M, D)``, computes
    the cosine similarity ``(N, K, M)``, masks gallery slots that are not
    yet filled (per the count field), reduces over the ``K`` slots, and
    returns ``1 - similarity`` as an ``(N, M)`` cost.

    Parameters
    ----------
    gallery_field : str
        Tracklet field holding the ``(N, K, D)`` gallery buffer.
    count_field : str
        Tracklet field holding the ``(N,)`` fill count (number of appends).
    field : str
        Detection field holding the ``(M, D)`` query embedding.
    reduce : {"max", "mean"}
        Slot reduction. ``"max"`` takes the best-matching historical view
        (robust to appearance change); ``"mean"`` averages valid slots.
    eps : float
        Lower bound on the L2 norm used for normalisation.

    Raises
    ------
    ValueError
        If ``reduce`` is not ``"max"`` or ``"mean"``.

    """

    gallery_field: str
    count_field: str
    field: str
    reduce: GalleryReduce = "max"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        """Reject an unknown reduction."""
        if self.reduce not in ("max", "mean"):
            msg = f"GalleryCost.reduce must be 'max' or 'mean'; got {self.reduce!r}"
            raise ValueError(msg)

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """Compute the gallery-reduced cosine cost matrix."""
        del ctx
        gallery = getattr(cs, self.gallery_field)  # (N, K, D)
        count = getattr(cs, self.count_field)  # (N,)
        query = getattr(ds, self.field)  # (M, D)
        n, k, _ = gallery.shape
        m = query.shape[0]

        g_norm = torch.nn.functional.normalize(gallery, dim=-1, eps=self.eps)
        q_norm = torch.nn.functional.normalize(query, dim=-1, eps=self.eps)
        sim = torch.einsum("nkd,md->nkm", g_norm, q_norm)  # (N, K, M)

        # Mask gallery slots that have not been filled yet.
        valid = (
            torch.arange(k, device=gallery.device)[None, :]
            < count.clamp(max=k)[:, None]
        )  # (N, K)
        valid_km = valid[:, :, None].expand(n, k, m)

        if self.reduce == "max":
            neg_inf = torch.finfo(sim.dtype).min
            masked = torch.where(valid_km, sim, torch.full_like(sim, neg_inf))
            reduced = masked.amax(dim=1)  # (N, M)
        else:
            denom = valid.sum(dim=1).clamp_min(1)[:, None]  # (N, 1)
            reduced = (sim * valid_km).sum(dim=1) / denom  # (N, M)

        return CostExpression.from_matrix(1.0 - reduced)
