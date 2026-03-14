"""
Learned MOTR-style appearance-filter modules.

The ``learned`` tracker (``build_learned_tracker`` in ``tracker.py``) replaces the
closed-form ``Identity``/``Replace`` embedding filter with a pair of small learned
modules wrapped by ``LearnedProcess`` / ``LearnedObservation``:

- :class:`Propagator` is the predict step — a residual MLP that nudges a track
  embedding forward in time and renormalizes it onto the unit sphere.
- :class:`Fuser` is the update step — a gated residual fuse of a track embedding
  with its matched detection's embedding.

Both are autograd-native, so the same cosine/Sinkhorn association objective used
at inference can train them (see ``train_learned.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

__all__ = ["Fuser", "Propagator"]


class Propagator(nn.Module):
    """Predict step: residual MLP over a track embedding, renormalized."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        """Build a ``dim -> hidden -> dim`` residual MLP."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Propagate ``x`` ``(N, D)`` one step and renormalize to unit norm."""
        del dt  # constant-rate propagation; dt accepted for the LearnedProcess hook
        return F.normalize(x + self.net(x), dim=-1)


class Fuser(nn.Module):
    """Update step: gated residual fuse of track + matched measurement."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        """Build a gate MLP over ``[track, measurement]`` (``2*dim -> dim``)."""
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, track: torch.Tensor, meas: torch.Tensor) -> torch.Tensor:
        """Fuse matched ``track``/``meas`` ``(K, D)`` pairs, renormalized."""
        g = self.gate(torch.cat([track, meas], dim=-1))
        return F.normalize(g * meas + (1 - g) * track, dim=-1)
