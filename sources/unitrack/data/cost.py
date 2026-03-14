"""Cost matrix with un-applied gate attachments."""

from __future__ import annotations

import torch
from tensordict import tensorclass

__all__ = ["CostExpression"]


@tensorclass
class CostExpression:
    """
    A cost matrix plus optional un-applied gates and bias.

    Gates are carried separately from the matrix so a downstream node
    (Merge, Associate) can apply them at the right moment. The upstream
    cost producer never has to materialize an ``(N, M)`` ``inf``-mask
    just to express a feasibility constraint.

    Attributes
    ----------
    matrix : torch.Tensor
        Float ``(N, M)`` raw cost matrix. Lower is a better match.
    gate_pair : torch.Tensor or None
        Optional bool ``(N, M)`` per-pair feasibility mask. ``True`` is
        allowed; ``False`` blocks the pair at materialisation.
    gate_cs : torch.Tensor or None
        Optional bool ``(N,)`` per-tracklet feasibility mask. ``False``
        rows become ``+inf`` at materialisation.
    gate_ds : torch.Tensor or None
        Optional bool ``(M,)`` per-detection feasibility mask. ``False``
        columns become ``+inf`` at materialisation.
    bias : torch.Tensor or None
        Optional float ``(N, M)`` additive bias applied on materialisation.

    """

    matrix: torch.Tensor
    gate_pair: torch.Tensor | None = None
    gate_cs: torch.Tensor | None = None
    gate_ds: torch.Tensor | None = None
    bias: torch.Tensor | None = None

    @classmethod
    def from_matrix(
        cls,
        matrix: torch.Tensor,
        *,
        gate_pair: torch.Tensor | None = None,
        gate_cs: torch.Tensor | None = None,
        gate_ds: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> CostExpression:
        """
        Build a :class:`~unitrack.data.CostExpression` from a matrix and optional gates.

        Parameters
        ----------
        matrix
            Float ``(N, M)`` cost matrix.
        gate_pair
            Optional bool ``(N, M)`` per-pair feasibility mask.
        gate_cs
            Optional bool ``(N,)`` per-tracklet feasibility mask.
        gate_ds
            Optional bool ``(M,)`` per-detection feasibility mask.
        bias
            Optional float ``(N, M)`` additive bias.

        Returns
        -------
        CostExpression
            The packed cost expression.

        Raises
        ------
        ValueError
            If ``matrix`` is not 2-D or if any gate/bias shape disagrees
            with ``matrix``.

        """
        # Validate shapes up-front so misuse fails here, not as a confusing
        # broadcast/index error deep inside :meth:`materialize`.
        if matrix.dim() != 2:
            msg = (
                "CostExpression.matrix must be 2-D (N, M); "
                f"got shape {tuple(matrix.shape)}"
            )
            raise ValueError(msg)
        n, m = matrix.shape
        if gate_pair is not None and tuple(gate_pair.shape) != (n, m):
            msg = (
                f"CostExpression.gate_pair shape {tuple(gate_pair.shape)} "
                f"does not match matrix shape {(n, m)}"
            )
            raise ValueError(msg)
        if gate_cs is not None and tuple(gate_cs.shape) != (n,):
            msg = (
                f"CostExpression.gate_cs shape {tuple(gate_cs.shape)} "
                f"does not match expected ({n},)"
            )
            raise ValueError(msg)
        if gate_ds is not None and tuple(gate_ds.shape) != (m,):
            msg = (
                f"CostExpression.gate_ds shape {tuple(gate_ds.shape)} "
                f"does not match expected ({m},)"
            )
            raise ValueError(msg)
        if bias is not None and tuple(bias.shape) != (n, m):
            msg = (
                f"CostExpression.bias shape {tuple(bias.shape)} "
                f"does not match matrix shape {(n, m)}"
            )
            raise ValueError(msg)
        return cls(
            matrix=matrix,
            gate_pair=gate_pair,
            gate_cs=gate_cs,
            gate_ds=gate_ds,
            bias=bias,
            batch_size=[],  # type: ignore[unknown-argument]
        )

    def materialize(self) -> torch.Tensor:
        """
        Return the cost matrix with all attached gates and bias applied.

        Returns
        -------
        torch.Tensor
            Float ``(N, M)`` materialised cost. Blocked pairs become
            ``+inf``; ``bias`` is added to the surviving entries.

        """
        out = self.matrix.clone()
        if self.bias is not None:
            out = out + self.bias
        if self.gate_pair is not None:
            out = torch.where(self.gate_pair, out, torch.full_like(out, float("inf")))
        if self.gate_cs is not None:
            blocked_cs = ~self.gate_cs
            out[blocked_cs, :] = float("inf")
        if self.gate_ds is not None:
            blocked_ds = ~self.gate_ds
            out[:, blocked_ds] = float("inf")
        return out
