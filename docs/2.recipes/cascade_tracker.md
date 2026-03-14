# Recipe: two-stage cascade tracker

A cascade tracker matches high-confidence detections by one cue and the
remainder by another. This recipe matches high-score instances by
appearance (cosine over embeddings) first, then matches the residual plus
low-score instances by mask-IoU — the appearance cue is trusted where the
detector is confident, with overlap as the fallback.

It maps onto 2.0 as a `Sequential` of two `Filter` stages. Each `Filter`
splits the detections on a score predicate (`on="ds"`) and routes its slice
into a class-gated `Pipe`; `Sequential` threads the residual of the first
stage into the second. `hi` is the cascade's internal score split,
independent of any `min_score` floor applied upstream.

```python
import torch

import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine, MaskIoU
from unitrack.data import TensorSpec
from unitrack.gates import ClassGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Filter, Gated, Pipe, Sequential
from unitrack.states import FromDetectionField, Identity, Replace, State

EMBED_DIM = 256


def build_cascade_tracker(
    *,
    height: int,
    width: int,
    hi: float = 0.5,
    cos_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    embed_dim: int = EMBED_DIM,
) -> unitrack.Tracker:
    # The class gate wraps each stage's Pipe (a per-pair gate biases a cost,
    # so it must sit on the Pipe, not on the Sequential cascade).
    hi_stage = Filter(
        predicate=lambda ds: ds.score >= hi,
        then=Gated(
            gate=ClassGate("category"),
            then=Pipe(
                cost=Cosine("embedding"),
                assoc=Associate(Jonker(threshold=cos_threshold)),
            ),
        ),
        on="ds",
    )
    lo_stage = Filter(
        predicate=lambda ds: ds.score < hi,
        then=Gated(
            gate=ClassGate("category"),
            then=Pipe(
                cost=MaskIoU("mask"),
                assoc=Associate(Jonker(threshold=iou_threshold)),
            ),
        ),
        on="ds",
    )
    pipeline = Sequential([hi_stage, lo_stage])
    return unitrack.Tracker(
        root=pipeline,
        states={
            "embedding": State(
                schema=TensorSpec(shape=(embed_dim,), dtype=torch.float32),
                process=Identity("embedding"),
                observation=Replace("embedding"),
                init=FromDetectionField("embedding"),
            ),
            "mask": State(
                schema=TensorSpec(shape=(height, width), dtype=torch.bool),
                process=Identity("mask"),
                observation=Replace("mask"),
                init=FromDetectionField("mask"),
            ),
            "category": State(
                schema=TensorSpec(shape=(), dtype=torch.int64),
                process=Identity("category"),
                observation=Replace("category"),
                init=FromDetectionField("category"),
            ),
            "score": State(
                schema=TensorSpec(shape=(), dtype=torch.float32),
                process=Identity("score"),
                observation=Replace("score"),
                init=FromDetectionField("score"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )
```

Detection requirements:

- `embedding` — `(M, embed_dim)` appearance vectors for the high-score stage.
- `mask` — `(M, height, width)` boolean instance masks for the low-score stage.
- `category` — `(M,)` int64 semantic class, for the `ClassGate` in each stage.
- `score` — `(M,)` float32 confidence, used by the `Filter` split predicates.
