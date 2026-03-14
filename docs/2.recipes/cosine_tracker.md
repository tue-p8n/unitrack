# Recipe: cosine appearance tracker

An appearance tracker matches instances frame-to-frame by the cosine
distance between per-instance embeddings, rather than by spatial overlap.
It maps onto 2.0 as a single `Pipe` whose cost is `Cosine` over an
`embedding` field, gated by class so only same-class detections compete.

Each detection must carry an L2-comparable embedding (here 256-d, the
dimensionality of a Mask2Former decoder query); the tracklet keeps the
last observed embedding (`Identity` predict, `Replace` on match). A match
requires `1 - cosine_similarity <= cost_threshold` within the same class.

```python
import torch

import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import TensorSpec
from unitrack.gates import ClassGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Gated, Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State

EMBED_DIM = 256


def build_cosine_tracker(
    *,
    cost_threshold: float = 0.5,
    embed_dim: int = EMBED_DIM,
) -> unitrack.Tracker:
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
                process=Identity("embedding"),
                observation=Replace("embedding"),
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

Detection requirements:

- `embedding` — `(M, embed_dim)` per-instance appearance vectors. `Cosine`
  L2-normalizes internally, so the raw decoder-query rows are fine.
- `category` — `(M,)` int64 semantic class, for the `ClassGate`.

An instance that lacks an embedding on a frame (e.g. a heavily-occluded
mask dropped by the model's post-processing) simply is not matched that
frame, like any other miss.
