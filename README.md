# Unified Tracking in PyTorch

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" alt="unitrack" width="340">
  </picture>
</p>

`unitrack` is a PyTorch-native multi-object tracking library for researchers
and ML engineers who want to assemble, not reimplement, a tracker. The library
decomposes the SORT-family pipeline into typed primitives (stages, costs, gates,
lifecycle policies, state recipes). Classical IoU+Kalman trackers, differentiable
soft-assignment variants, and cascaded or parallel fusions all share the same
scaffolding and run on a single `Tracker` core. Detections, tracklets, and frame
context flow through the pipeline as structured `TensorDict` records, enabling
multi-stream batching, clip-based inference, and `torch.func.vmap` compatibility
without separate code paths.

## Installation

- `python >= 3.13, < 3.14`
- `torch >= 2.7`

```bash
pip install unitrack
```

See [`docs/installation.md`](docs/installation.md) for the in-tree LAP solver
extras and the development workflow.

## Composable primitives

Each top-level subpackage covers one axis of the tracker:

- `unitrack.assignment` — solvers for the linear assignment problem
  (Hungarian, Greedy, Auction, Jonker–Volgenant variants) plus soft companions
  for differentiable matching.
- `unitrack.costs` — pairwise cost producers (cosine, L2, IoU family,
  Mahalanobis) and combinators (`Reduce`, `Weighted`, `Sinkhorn`).
- `unitrack.data` — the typed records that flow through the pipeline:
  `Detections`, `Tracklets`, `FrameContext`, `CostExpression`, `MatchOutcome`,
  plus clip-shaped counterparts and the `Gate` algebra.
- `unitrack.gates` — class, score, spatial, and motion gates that mask
  candidate pairs before cost computation.
- `unitrack.lifecycle` — tracklet status policies (`StandardLifecycle`,
  `ConfirmedOnly`, `NoLifecycle`) and visibility filters.
- `unitrack.pipeline` — the stage tree: `Pipe`, `Sequential`, `Parallel`,
  `Gated`, `Filter`, `Iterate`.
- `unitrack.states` — per-feature state recipes (`Identity`, `Replace`, `EMA`)
  and the Kalman family (`KalmanBBox`, `KalmanCentroid`, `KalmanUpdate`).
- `unitrack.tracker` — the top-level `Tracker` module and its wrappers:
  `MultiStream`, `BatchTracker`, `ClipTracker`, `TrackletMemory`.

## Usage

This example tracks objects across a three-frame sequence using a distance cost
and the Jonker–Volgenant solver.

```python
import torch
import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import CDist
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State

tracker = unitrack.Tracker(
    root=Pipe(cost=CDist("position"),
              assoc=Associate(Jonker(threshold=10))),
    states={
        "position": State(
            schema=TensorSpec(shape=(1,), dtype=torch.float32),
            process=Identity("position"),
            observation=Replace("position"),
            init=FromDetectionField("position"),
        ),
    },
    lifecycle=NoLifecycle(),
    visibility=IncludeAll(),
)
ms = unitrack.MultiStream(tracker)

for frame in range(3):
    n = 1 + frame * 2
    pos = (torch.arange(n, dtype=torch.float32) + 1.0).unsqueeze(1)
    ds = Detections(index=torch.arange(n, dtype=torch.int64),
                     position=pos, batch_size=[n])
    res = ms.step(0, ds, FrameContext.make(frame, fps=15.0, stream_key=0))
    print(f"frame {frame}: ids={res.ids.tolist()}")
```

## Recipes and tutorials

Ready-to-run tracker recipes in [`docs/recipes/`](docs/recipes/) include a SORT
port (IoU + Kalman bounding-box state) and a class-and-score-gated overlap
tracker. The six-notebook tour under
[`notebooks/tutorials/`](notebooks/tutorials/) covers the data model,
cost and gate zoos, pipeline tree, state and lifecycle, and cascaded
versus parallel fusion on synthetic data with known ground truth.
