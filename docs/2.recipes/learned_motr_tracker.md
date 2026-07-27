# Recipe: learned MOTR-style appearance tracker

The learned tracker is the cosine appearance tracker with its closed-form
embedding filter replaced by two small *learned* modules, in the spirit of
MOTR's query-propagation filter:

- a **Propagator** — the predict step: a residual MLP that nudges a track
  embedding forward in time and renormalizes it onto the unit sphere.
- a **Fuser** — the update step: a gated residual fuse of a track embedding
  with its matched detection's embedding.

`LearnedProcess` wraps the Propagator into the state's predict, and
`LearnedObservation` wraps the Fuser into the state's update on match. The
association is unchanged from `cosine_tracker.md`: `Cosine("embedding")`
gated by class. Because both modules are autograd-native, the same
cosine/Sinkhorn association objective used at inference trains them.

The two modules are trained once by
`sources/unitrack/benchmarks/hota/train_learned.py`
(`python -m unitrack.benchmarks.hota.train_learned`), which extracts detection
embeddings over a few train clips, assigns each a GT track id by mask-IoU, and
optimizes the propagated-then-matched embeddings to recover the GT
correspondences via a Sinkhorn soft-assignment loss. It writes a small
`safetensors` checkpoint that this factory loads; the factory raises a clear
`FileNotFoundError` pointing at the training script if the checkpoint is
absent, so the filter is never silently untrained.

```python
import torch
import torch.nn.functional as F
from torch import nn

import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import TensorSpec
from unitrack.gates import ClassGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Gated, Pipe
from unitrack.states import (
    FromDetectionField,
    Identity,
    LearnedObservation,
    LearnedProcess,
    Replace,
    State,
)

EMBED_DIM = 256


class Propagator(nn.Module):
    """Predict step: residual MLP over a track embedding, renormalized."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        del dt  # constant-rate propagation
        return F.normalize(x + self.net(x), dim=-1)


class Fuser(nn.Module):
    """Update step: gated residual fuse of track + matched measurement."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, track: torch.Tensor, meas: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([track, meas], dim=-1))
        return F.normalize(g * meas + (1 - g) * track, dim=-1)


def build_learned_tracker(
    *,
    checkpoint: str,
    cost_threshold: float = 0.5,
    embed_dim: int = EMBED_DIM,
) -> unitrack.Tracker:
    from safetensors.torch import load_file

    flat = load_file(checkpoint)  # raises if the checkpoint is missing
    prop, fuse = Propagator(embed_dim), Fuser(embed_dim)
    prop.load_state_dict(
        {
            k[len("propagator.") :]: v
            for k, v in flat.items()
            if k.startswith("propagator.")
        }
    )
    fuse.load_state_dict(
        {k[len("fuser.") :]: v for k, v in flat.items() if k.startswith("fuser.")}
    )
    prop.eval()
    fuse.eval()

    inner = Pipe(
        cost=Cosine("embedding"),
        assoc=Associate(Jonker(threshold=cost_threshold)),
    )
    pipeline = Gated(gate=ClassGate("category"), then=inner)
    return unitrack.Tracker(
        root=pipeline,
        states={
            "embedding": State(
                schema=TensorSpec(shape=(embed_dim,), dtype=torch.float32),
                process=LearnedProcess("embedding", prop),
                observation=LearnedObservation("embedding", "embedding", fuse),
                init=FromDetectionField("embedding"),
            ),
            "category": State(
                schema=TensorSpec(shape=(), dtype=torch.int64),
                process=Identity("category"),
                observation=Replace("category"),
                init=FromDetectionField("category"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )
```

The packaged factory (`unitrack.benchmarks.hota.tracker.build_learned_tracker`)
defaults `checkpoint` to the committed
`benchmarks/hota/weights/learned_filter.safetensors`.

Detection requirements: identical to `cosine_tracker.md` — an `embedding`
`(M, embed_dim)` per instance and an int64 `category` `(M,)`.
