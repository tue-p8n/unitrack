"""Distance-based :class:`~unitrack.pipeline.CostProducer` leaves."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import CostExpression, Detections, FrameContext, Tracklets
from unitrack.states.kalman.project import mahalanobis_d2

__all__ = ["RBF", "BiSoftmax", "CDist", "Chamfer", "Cosine", "Mahalanobis"]


def _get_field(obj, name: str) -> torch.Tensor:
    """Read a (possibly user-supplied) field from a snapshot, raising on missing."""
    if not hasattr(obj, name):
        msg = f"required field {name!r} not present on {type(obj).__name__}"
        raise KeyError(msg)
    return getattr(obj, name)


@dataclasses.dataclass(frozen=True, slots=True)
class Cosine:
    """
    ``1 - cosine_similarity`` cost over a named float-vector field.

    Attributes
    ----------
    field : str
        Name of the float-vector field on both ``cs`` and ``ds``.
    eps : float
        Lower bound on the L2 norm used for normalisation, guarding
        against zero-vector inputs.

    """

    field: str
    eps: float = 1e-5

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the cosine-distance cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix; lower is better.

        """
        del ctx
        a = _get_field(cs, self.field)
        b = _get_field(ds, self.field)
        a_norm = a / a.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        b_norm = b / b.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        sim = a_norm @ b_norm.mT
        return CostExpression.from_matrix(1.0 - sim)


@dataclasses.dataclass(frozen=True, slots=True)
class CDist:
    """
    ``‖a - b‖_p`` distance over a named float-vector field.

    Attributes
    ----------
    field : str
        Name of the float-vector field on both ``cs`` and ``ds``.
    p_norm : float
        Order of the norm; defaults to Euclidean (``p = 2``).

    """

    field: str
    p_norm: float = 2.0

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the pairwise :math:`L_p` distance cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix; lower is better.

        """
        del ctx
        a = _get_field(cs, self.field)
        b = _get_field(ds, self.field)
        return CostExpression.from_matrix(
            torch.cdist(
                a, b, p=self.p_norm, compute_mode="donot_use_mm_for_euclid_dist"
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class BiSoftmax:
    """
    Bi-directional softmax cost ``1 - 0.5 * (softmax_cs + softmax_ds)``.

    The inner ``sim = a @ b.mT`` is reduced via a softmax along each
    axis; the symmetric average is then inverted into a distance.

    Attributes
    ----------
    field : str
        Name of the float-vector field on both ``cs`` and ``ds``.

    """

    field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the bi-directional softmax cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix; lower is better.

        """
        del ctx
        a = _get_field(cs, self.field)
        b = _get_field(ds, self.field)
        sim = a @ b.mT
        a2b = sim.softmax(dim=0)
        b2a = sim.softmax(dim=1)
        return CostExpression.from_matrix(1.0 - 0.5 * (a2b + b2a))


@dataclasses.dataclass(frozen=True, slots=True)
class RBF:
    """
    RBF-kernel cost ``1 - exp(-gamma * ‖a - b‖²)``.

    Attributes
    ----------
    field : str
        Name of the float-vector field on both ``cs`` and ``ds``.
    gamma : float
        Kernel bandwidth; larger values sharpen the decay.

    """

    field: str
    gamma: float = 1.0

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the RBF-kernel cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix bounded in ``[0, 1]``; lower is better.

        """
        del ctx
        a = _get_field(cs, self.field)
        b = _get_field(ds, self.field)
        return CostExpression.from_matrix(
            1.0 - torch.exp(-self.gamma * torch.cdist(a, b, p=2.0) ** 2),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Mahalanobis:
    """
    Mahalanobis squared distance ``(a - b)^T Σ^-1 (a - b)``.

    Σ is read from a Kalman state's covariance field on ``cs``.

    Attributes
    ----------
    field : str
        Name of the mean-vector field shared by ``cs`` and ``ds``.
    cov_field : str
        Name of the covariance field on ``cs``.

    """

    field: str
    cov_field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the Mahalanobis squared-distance cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows; must carry both
            :attr:`field` and :attr:`cov_field`.
        ds
            Detection record with ``M`` rows; must carry :attr:`field`.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix; lower is better.

        """
        del ctx
        a = _get_field(cs, self.field)
        cov = _get_field(cs, self.cov_field)
        b = _get_field(ds, self.field)
        return CostExpression.from_matrix(
            mahalanobis_d2(a, cov, b, label="Mahalanobis")
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Chamfer:
    """
    Symmetric chamfer distance over a fixed-size point-cloud field.

    The field shape must be ``(B, K, D)`` where ``B`` is the
    tracklet/detection batch, ``K`` is points per cloud, and ``D`` is
    point dimensionality.

    Attributes
    ----------
    field : str
        Name of the point-cloud field on both ``cs`` and ``ds``.

    """

    field: str

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        ctx: FrameContext,
    ) -> CostExpression:
        """
        Compute the symmetric chamfer-distance cost matrix.

        Parameters
        ----------
        cs
            Tracklet snapshot with ``N`` rows.
        ds
            Detection record with ``M`` rows.
        ctx
            Frame context (unused).

        Returns
        -------
        CostExpression
            ``(N, M)`` cost matrix; lower is better.

        """
        del ctx
        a = _get_field(cs, self.field)  # (N, K, D)
        b = _get_field(ds, self.field)  # (M, K, D)
        # (N, M, K_a, K_b) pairwise squared L2.
        diffs = a[:, None, :, None, :] - b[None, :, None, :, :]
        sq = (diffs**2).sum(dim=-1)
        a_to_b = sq.min(dim=-1).values.mean(dim=-1)  # (N, M)
        b_to_a = sq.min(dim=-2).values.mean(dim=-1)  # (N, M)
        return CostExpression.from_matrix(a_to_b + b_to_a)
