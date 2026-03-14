# Recipe: Kalman motion tracker

A motion tracker associates detections to the *predicted* position of each
tracklet, rejecting spatially-implausible matches. This recipe runs a
constant-velocity Kalman filter on instance centroids: the tracklet state is
4-D `[x, y, vx, vy]`, advanced one step each frame; the detection carries a
2-D `[x, y]` centroid measurement.

`KalmanCentroid2D.state_entries` seeds the `centroid` (predicted mean) and
`centroid_cov` (covariance) tracklet fields and supplies the predict/update.
The association cost is the **Mahalanobis** chi-squared distance from each
detection centroid to a tracklet's predicted centroid. A plain
`CDist("centroid")` cannot be used here: the tracklet mean is 4-D while the
detection mean is 2-D, so the feature dimensions mismatch. `Mahalanobis`
is the primitive that projects the 4-D state into the 2-D measurement
subspace using `centroid_cov`. The same metric drives the `MotionGate`,
which rejects matches whose chi-squared distance exceeds `max_chi2`
(`5.9915` is the 0.95 quantile at 2 d.o.f. — the standard SORT/DeepSORT
motion gate), and a `ClassGate` keeps matching within a class.

`q` / `r` are the process / measurement noise; they are sized for
pixel-space centroids (order one pixel²), not the normalized-coordinate
defaults of `KalmanCentroid2D`.

```python
import torch

import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Mahalanobis
from unitrack.data import TensorSpec
from unitrack.gates import ClassGate, MotionGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Gated, Pipe, Sequential
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.states.kalman import KalmanCentroid2D

# 0.95 quantile of chi-squared at 2 d.o.f. (centroid x, y).
CHI2_GATE = 5.9915


def build_kalman_tracker(
    *,
    max_chi2: float = CHI2_GATE,
    q: float = 1.0,
    r: float = 1.0,
) -> unitrack.Tracker:
    kal = KalmanCentroid2D(field="centroid", q=q, r=r)
    centroid_states = kal.state_entries(meas_field="centroid")
    gate = Sequential(
        [
            ClassGate("category"),
            MotionGate(
                mean_field="centroid",
                cov_field="centroid_cov",
                max_chi2=max_chi2,
            ),
        ]
    )
    inner = Pipe(
        cost=Mahalanobis("centroid", "centroid_cov"),
        assoc=Associate(Jonker(threshold=max_chi2)),
    )
    pipeline = Gated(gate=gate, then=inner)
    return unitrack.Tracker(
        root=pipeline,
        states={
            **centroid_states,
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

- `centroid` — `(M, 2)` `[x, y]` pixel centroid per instance (e.g. the mask
  pixel-mean). `KalmanCentroid2D` lifts it to the 4-D position/velocity state.
- `category` — `(M,)` int64 semantic class, for the `ClassGate`.

The `Jonker` `threshold` is set to `max_chi2` so the cost matrix and the gate
agree on the maximum acceptable Mahalanobis distance; the `MotionGate` does
the principled per-pair rejection.
