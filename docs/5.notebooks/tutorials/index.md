# Tutorial notebooks

A seven-notebook tour of unitrack 2.0.
Each notebook is self-contained;
read in order if new to the library, or jump to a specific topic.

| # | Notebook | Topic |
|---|---|---|
| 1 | [Quickstart](/notebooks/tutorials/quickstart) | Build & run a minimal tracker. Visualize trajectories. |
| 2 | [Data model](/notebooks/tutorials/data_model) | The five typed records: `Tracklets`, `Detections`, `FrameContext`, `CostExpression`, `MatchOutcome`, plus the `Gate` algebraic variant. |
| 3 | [Costs and gates](/notebooks/tutorials/costs_and_gates) | The full cost zoo (Cosine, CDist, BiSoftmax, RBF, IoU family, Mahalanobis) and gate zoo (Class, Score, Spatial, Motion). With heatmaps. |
| 4 | [Pipeline tree](/notebooks/tutorials/pipeline_tree) | Composable stage tree: `Pipe`, `Sequential`, `Parallel`, `Gated`, `Filter`, `Iterate`. Cascaded vs parallel fusion. |
| 5 | [States and lifecycle](/notebooks/tutorials/states_and_lifecycle) | State evolution (Process × Observation), Kalman, EMA. Lifecycle (Tentative → Active → Lost → Removed). |
| 6 | [Cascaded and parallel](/notebooks/tutorials/cascaded_and_parallel) | K=2 cascaded and parallel-fusion configurations compared end-to-end on synthetic data with known ground truth. |
| 7 | [Migration](/notebooks/tutorials/migration) | Migrate a 1.x tracker to 2.0 and showcase the new possibilities (parallel fusion, cascaded matching, lifecycle, differentiable matching) — driven by a real pretrained detector. |

Notebooks are generated from `_build.py` in this directory;
edit the `NB_*` cell lists there, re-run it, then execute the
notebook to refresh outputs. Edit the lists, not the `.ipynb`
files directly — regenerating overwrites them, outputs included.

## What's not covered

The tutorials focus on the core library.
For these specialized paths, look at:

- **Multi-stream / batched tracking** — `unitrack.tracker.BatchTracker`
  (vmap-batched).
  See its tests in `tests/unitrack/tracker/test_batch_tracker.py`.
- **Clip-based inference** (MinVIS, DVIS++ patterns) —
  `unitrack.tracker.ClipTracker`.
- **Differentiable tracking** — `Tracker(differentiable=True)` swaps in
  soft companions automatically.
- **Real Optuna sweep on Mask2Former-Cityscapes detections** —
  `examples/hpo_sweep/`.
