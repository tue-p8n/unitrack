---
hero:
  variant: centered
  specs:
    - PyTorch-native
    - SORT-family pipeline
    - Composable primitives
---

# Unitrack

`unitrack` is a PyTorch-native multi-object tracking library that exposes the
SORT-family pipeline as a small set of composable primitives: assignment
solvers, pairwise cost producers, gates, per-feature state recipes, lifecycle
policies, and a typed stage tree that wires them together. Classical
IoU+Kalman trackers, soft-assignment differentiable variants, and cascaded or
parallel fusions all share the same `Tracker` core and the same typed records
(`Detections`, `Tracklets`, `FrameContext`).

The documentation is organized for two reading orders. New users should start
with [Getting Started](/getting-started), which covers installation and then
walks through the tracker recipes and the tutorial notebooks. Readers who
already know the library can jump straight to the [API reference](/api)
for the per-subpackage symbol lists.

## Start here

1. **[Getting Started](/getting-started)**: install, pick a solver backend, and
   find the recipes and tutorial notebooks.
2. **[Migration](/migration)**: mapping the 1.x surface onto 2.0.
3. **[API Reference](/api)**: the full, generated symbol reference.
