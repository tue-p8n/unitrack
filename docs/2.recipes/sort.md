# Recipe: SORT-style tracker on `unitrack` 2.0

SORT (Bewley et al., 2016) is a single-stage IoU + Kalman bbox tracker.
It maps onto 2.0 as one `Pipe` whose cost is `BoxIoU` and whose state
includes a `KalmanBBox` predict + `KalmanUpdate` observation.

SORT's tracklet state is the 7-D ``[x, y, a, h, vx, vy, va]`` constant-velocity
model — `x, y` is the box centre, `a` is aspect ratio, `h` is height. The
*measurement* is the 4-D leading slice ``[x, y, a, h]``: detections must
therefore carry a ``bbox_xyah`` field of shape ``(M, 4)`` in that order
(convert from xyxy upstream — ``PadZerosInitializer`` zero-pads it to
the full 7-D state at spawn). The IoU cost runs on ``bbox_xyxy``, so the
tracklet must also expose a ``bbox_xyxy`` projection of the predicted
Kalman mean — `KalmanBBox` only seeds ``bbox_xyah``/``bbox_xyah_cov``,
so the recipe adds a small projection state that derives ``bbox_xyxy``
from ``bbox_xyah`` after each Kalman predict.

```python
import dataclasses

import torch

import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import BoxIoU
from unitrack.data import (
    Detections,
    FrameContext,
    MatchOutcome,
    TensorSpec,
    Tracklets,
)
from unitrack.lifecycle import ConfirmedOnly, StandardLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, State
from unitrack.states.kalman import KalmanBBox


def _xyah_to_xyxy(xyah: torch.Tensor) -> torch.Tensor:
    cx, cy, a, h = xyah[..., 0], xyah[..., 1], xyah[..., 2], xyah[..., 3]
    w = a * h
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


@dataclasses.dataclass(frozen=True, slots=True)
class _ProjectXyahToXyxy:
    """Derive `bbox_xyxy` from the (already-predicted) `bbox_xyah` mean."""

    source: str = "bbox_xyah"
    target: str = "bbox_xyxy"

    def __call__(self, cs: Tracklets, ctx: FrameContext) -> Tracklets:
        del ctx
        if cs.batch_size[0] == 0:
            return cs
        return cs.set(self.target, _xyah_to_xyxy(getattr(cs, self.source)))


@dataclasses.dataclass(frozen=True, slots=True)
class _ProjectXyahToXyxyAfterUpdate:
    """Re-derive `bbox_xyxy` after the Kalman update wrote a new mean."""

    source: str = "bbox_xyah"
    target: str = "bbox_xyxy"

    def __call__(
        self,
        cs: Tracklets,
        ds: Detections,
        match: MatchOutcome,
        ctx: FrameContext,
    ) -> Tracklets:
        del ds, match, ctx
        if cs.batch_size[0] == 0:
            return cs
        return cs.set(self.target, _xyah_to_xyxy(getattr(cs, self.source)))


def build_sort_tracker(
    *,
    iou_threshold: float = 0.3,
    min_hits: int = 3,
    max_age: int = 1,
) -> unitrack.Tracker:
    bbox = KalmanBBox("bbox_xyah")
    states = {
        # KalmanBBox seeds bbox_xyah + bbox_xyah_cov; predict advances both.
        **bbox.state_entries(),
        # Projection state: derives bbox_xyxy from the Kalman mean *after*
        # the predict step (the iteration order in `states` decides this:
        # bbox_xyah's predict runs first, then this projection writes the
        # corresponding bbox_xyxy onto the snapshot). Initialized directly
        # from the detection's bbox_xyxy so first-frame tracklets have a
        # consistent xyxy field before any predict has run.
        "bbox_xyxy": State(
            schema=TensorSpec(shape=(4,), dtype=torch.float32),
            process=_ProjectXyahToXyxy(),
            observation=_ProjectXyahToXyxyAfterUpdate(),
            init=FromDetectionField("bbox_xyxy"),
        ),
    }
    return unitrack.Tracker(
        root=Pipe(
            cost=BoxIoU("bbox_xyxy"),
            assoc=Associate(Jonker(threshold=1.0 - iou_threshold)),
        ),
        states=states,
        lifecycle=StandardLifecycle(min_hits=min_hits, max_age=max_age),
        visibility=ConfirmedOnly(),
    )
```

Detection requirements:

- ``bbox_xyah`` — ``(M, 4)`` centre/aspect/height for the Kalman update.
- ``bbox_xyxy`` — ``(M, 4)`` xyxy for the IoU cost and as the seed for
  the projection state above. The two must agree (use a single source of
  truth in your detector adapter; ``_xyah_to_xyxy`` is one direction).
