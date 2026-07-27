# Recipe: overlap-IoU tracker (port of 1.x `models.overlap`)

The 1.x `unitrack.models.build_overlap_tracker` is a single-stage class +
score-gated IoU matcher. The 2.0 port is a short recipe:

```python
import torch
import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import BoxCIoU
from unitrack.data import TensorSpec
from unitrack.gates import ClassGate, ScoreGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Gated, Pipe, Sequential
from unitrack.states import FromDetectionField, Identity, Replace, State


def build_overlap_tracker(
    *,
    threshold: float = 0.5,
    min_score: float = 0.1,
    class_gate: bool = True,
) -> unitrack.Tracker:
    cost = BoxCIoU("bbox")
    inner = Pipe(cost=cost, assoc=Associate(Jonker(threshold=threshold)))
    gates = [ScoreGate("score", threshold=min_score)]
    if class_gate:
        gates.insert(0, ClassGate("klass"))
    pipeline = Gated(gate=Sequential(gates), then=inner)
    return unitrack.Tracker(
        root=pipeline,
        states={
            "bbox": State(
                schema=TensorSpec(shape=(4,), dtype=torch.float32),
                process=Identity("bbox"),
                observation=Replace("bbox"),
                init=FromDetectionField("bbox"),
            ),
            "score": State(
                schema=TensorSpec(shape=(), dtype=torch.float32),
                process=Identity("score"),
                observation=Replace("score"),
                init=FromDetectionField("score"),
            ),
            "klass": State(
                schema=TensorSpec(shape=(), dtype=torch.int64),
                process=Identity("klass"),
                observation=Replace("klass"),
                init=FromDetectionField("klass"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )
```
