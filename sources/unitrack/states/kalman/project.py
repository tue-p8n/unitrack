"""Measurement-projection and PSD-solve helpers used by Mahalanobis primitives."""

from __future__ import annotations

import torch

__all__ = ["mahalanobis_d2", "project_to_measurement", "solve_psd"]


def project_to_measurement(
    mean: torch.Tensor,
    cov: torch.Tensor,
    d_meas: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncate a Kalman state and covariance to the measurement subspace.

    Mirrors :class:`unitrack.gates.MotionGate`: when the state has higher
    dimension than the measurement (e.g. a 6-D CV state vs a 3-D centroid
    detection), the measurement matrix is taken as ``H = [I, 0]`` and the
    leading ``d_meas`` rows and columns are returned. When the dimensions
    already match the inputs are returned unchanged.

    Parameters
    ----------
    mean : torch.Tensor
        ``(..., D)`` state mean.
    cov : torch.Tensor
        ``(..., D, D)`` state covariance.
    d_meas : int
        Measurement subspace dimension. Must be ``<= D``.

    Returns
    -------
    mean_proj : torch.Tensor
        ``(..., d_meas)`` truncated mean.
    cov_proj : torch.Tensor
        ``(..., d_meas, d_meas)`` truncated covariance.

    Raises
    ------
    ValueError
        If ``d_meas`` is greater than the state dimension.

    """
    if mean.shape[-1] == d_meas:
        return mean, cov
    if mean.shape[-1] < d_meas:
        msg = (
            f"project_to_measurement: state dim {mean.shape[-1]} is smaller than "
            f"measurement dim {d_meas}"
        )
        raise ValueError(msg)
    return mean[..., :d_meas], cov[..., :d_meas, :d_meas]


def solve_psd(cov: torch.Tensor, rhs: torch.Tensor, *, label: str) -> torch.Tensor:
    """
    Solve ``cov @ x = rhs`` for a (batch of) PSD covariance(s).

    Parameters
    ----------
    cov : torch.Tensor
        ``(..., D, D)`` positive-semi-definite system matrix.
    rhs : torch.Tensor
        ``(..., D, K)`` right-hand side.
    label : str
        Identifier embedded in the error message on failure.

    Returns
    -------
    torch.Tensor
        ``(..., D, K)`` solution tensor.

    Raises
    ------
    RuntimeError
        If the solve produces non-finite values, indicating a singular
        covariance or PSD-cone drift. The caller should re-tune
        process/measurement noise or the initial covariance.

    Notes
    -----
    Not vmap- or ``torch.compile``-safe: the finiteness check is a
    host-side, data-dependent branch and forces a D2H sync on CUDA.

    """
    out = torch.linalg.solve(cov, rhs)
    if not torch.isfinite(out).all():
        msg = (
            f"{label}: covariance solve produced non-finite values "
            "(likely a singular covariance or PSD-cone drift)."
        )
        raise RuntimeError(msg)
    return out


def mahalanobis_d2(
    cs_field: torch.Tensor,
    cs_cov: torch.Tensor,
    ds_field: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    r"""
    Pairwise Mahalanobis chi-squared over a measurement subspace.

    Truncates the state to ``ds_field``'s dimension (``H = [I, 0]``),
    then solves :math:`\Sigma x = (a - b)` per pair and returns
    :math:`(a - b)^T x`. Shared by :class:`Mahalanobis`,
    :class:`~unitrack.gates.MotionGate`, and
    :class:`~unitrack.gates.SoftMotionGate` so they agree on
    the projection and PSD-guard conventions.

    Parameters
    ----------
    cs_field : torch.Tensor
        ``(N, D)`` tracklet mean.
    cs_cov : torch.Tensor
        ``(N, D, D)`` tracklet covariance.
    ds_field : torch.Tensor
        ``(M, d_meas)`` detection field.
    label : str
        Identifier embedded in :func:`solve_psd`'s error message.

    Returns
    -------
    torch.Tensor
        ``(N, M)`` pairwise chi-squared distances.

    """
    a, cov = project_to_measurement(cs_field, cs_cov, ds_field.shape[-1])
    diff = a[:, None, :] - ds_field[None, :, :]
    cov_b = cov[:, None].expand(-1, ds_field.shape[0], -1, -1)
    sol = solve_psd(cov_b, diff.unsqueeze(-1), label=label).squeeze(-1)
    return (diff * sol).sum(dim=-1)
