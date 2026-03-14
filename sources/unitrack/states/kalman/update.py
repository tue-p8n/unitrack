"""KalmanUpdate Observation — Joseph-form update; predict-only on miss."""

from __future__ import annotations

import dataclasses

import torch

from unitrack.data import Detections, FrameContext, MatchOutcome, Tracklets
from unitrack.states.kalman.project import solve_psd

__all__ = ["KalmanUpdate"]


@dataclasses.dataclass(frozen=True, slots=True)
class KalmanUpdate:
    """
    Kalman measurement update (Joseph form) for matched tracklet-detection pairs.

    Reads predicted mean ``cs.{field}`` of shape ``(N, D)`` and covariance
    ``cs.{cov_field}`` of shape ``(N, D, D)``, reads detection
    measurements ``ds.{field}`` of shape ``(M, M_dim)``, and writes back
    the updated mean and covariance for matched tracklets. Unmatched
    tracklets keep their predicted state (predict-only on miss).

    The update applies the standard Kalman gain
    ``K = P H^T (H P H^T + R)^{-1}`` via ``solve_psd``, then
    propagates covariance through the Joseph form
    ``(I - K H) P (I - K H)^T + K R K^T`` to preserve positive
    semi-definiteness under finite-precision arithmetic. The result is
    symmetrised to remove off-diagonal drift that accumulates over long
    sequences.

    ``H`` shape ``(M_dim, D)`` is the measurement matrix and ``R`` shape
    ``(M_dim, M_dim)`` is the measurement-noise covariance. Each
    bbox/centroid process ships a ``make_update(...)`` factory that
    constructs the matching :class:`KalmanUpdate`.

    Parameters
    ----------
    field : str
        Field name for the mean.
    cov_field : str
        Field name for the covariance.
    H : torch.Tensor
        ``(M_dim, D)`` measurement matrix.
    R : torch.Tensor
        ``(M_dim, M_dim)`` measurement-noise covariance.

    Raises
    ------
    ValueError
        If ``H`` and ``R`` have different dtypes (would force silent
        per-call casting).

    """

    field: str
    cov_field: str
    H: torch.Tensor
    R: torch.Tensor

    def __post_init__(self) -> None:
        """Reject mismatched H/R dtypes that would silently down-cast."""
        if self.H.dtype != self.R.dtype:
            msg = (
                f"KalmanUpdate: H and R must share a dtype to avoid silent "
                f"per-call casting; got H={self.H.dtype!r}, R={self.R.dtype!r}"
            )
            raise ValueError(msg)

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        """Apply Kalman measurement update for matched tracklet-detection pairs."""
        del ctx
        if match.matched_pairs.shape[0] == 0:
            return cs
        mean = getattr(cs, self.field).clone()
        cov = getattr(cs, self.cov_field).clone()
        cs_idx = match.matched_pairs[:, 0]
        ds_idx = match.matched_pairs[:, 1]
        z = getattr(ds, self.field)[ds_idx]  # (K, M)
        x = mean[cs_idx]  # (K, D)
        p = cov[cs_idx]  # (K, D, D)

        device = x.device
        h = self.H.to(device=device, dtype=x.dtype)
        r = self.R.to(device=device, dtype=x.dtype)
        y = z - x @ h.T  # (K, M)
        s = h @ p @ h.T + r  # (K, M, M)
        # gain = p h^T s^-1
        ht = h.T
        gain = solve_psd(s, (p @ ht).transpose(-1, -2), label="KalmanUpdate").transpose(
            -1, -2
        )

        x_new = x + (gain @ y.unsqueeze(-1)).squeeze(-1)
        # Joseph form: (I - K H) P (I - K H)^T + K R K^T. Preserves PSD of P
        # under finite-precision arithmetic where the standard form
        # (I - K H) P can drift into the indefinite cone.
        eye = torch.eye(p.shape[-1], device=p.device, dtype=p.dtype)
        kh = gain @ h
        i_minus_kh = eye - kh
        gain_t = gain.transpose(-1, -2)
        p_new = i_minus_kh @ p @ i_minus_kh.transpose(-1, -2) + gain @ r @ gain_t
        # Symmetrise to clean up float-precision asymmetry that accumulates
        # over many frames and can push covariance off the PSD cone.
        p_new = 0.5 * (p_new + p_new.transpose(-1, -2))

        mean[cs_idx] = x_new
        cov[cs_idx] = p_new
        return cs.set(self.field, mean).set(self.cov_field, cov)
