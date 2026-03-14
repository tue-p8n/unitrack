# unitrack tutorial notebooks

A six-notebook tour of unitrack 2.0. Each notebook is self-contained;
read in order if you're new to the library, or jump to a specific
topic.

| # | Notebook | Topic |
|---|---|---|
| 1 | [01_quickstart.ipynb](01_quickstart.ipynb) | Build & run a minimal tracker. Visualize trajectories. |
| 2 | [02_data_model.ipynb](02_data_model.ipynb) | The five typed records: `Tracklets`, `Detections`, `FrameContext`, `CostExpression`, `MatchOutcome`, plus the `Gate` algebraic variant. |
| 3 | [03_costs_and_gates.ipynb](03_costs_and_gates.ipynb) | The full cost zoo (Cosine, CDist, BiSoftmax, RBF, IoU family, Mahalanobis) and gate zoo (Class, Score, Spatial, Motion). With heatmaps. |
| 4 | [04_pipeline_tree.ipynb](04_pipeline_tree.ipynb) | Composable stage tree: `Pipe`, `Sequential`, `Parallel`, `Gated`, `Filter`, `Iterate`. Cascaded vs parallel fusion. |
| 5 | [05_states_and_lifecycle.ipynb](05_states_and_lifecycle.ipynb) | State evolution (Process × Observation), Kalman, EMA. Lifecycle (Tentative → Active → Lost → Removed). |
| 6 | [06_cascaded_and_parallel.ipynb](06_cascaded_and_parallel.ipynb) | K=2 cascaded and parallel-fusion configurations compared end-to-end on synthetic data with known ground truth. |
| 7 | [07_migration.ipynb](07_migration.ipynb) | Migrate a 1.x tracker to 2.0 and showcase the new possibilities (parallel fusion, cascaded matching, lifecycle, differentiable matching) — driven by a real pretrained detector. |

## Running them

Notebooks 1–6 need only `unitrack` and `matplotlib`:

```bash
uv pip install matplotlib
jupyter lab notebooks/tutorials/
```

Notebook 7 additionally uses a tiny Hugging Face detector and a
torchvision ReID encoder; install those (the notebook's first cell does
this too, and falls back to a synthetic seed if they are unavailable):

```bash
uv pip install matplotlib transformers torchvision pillow
```

## Regenerating

The notebooks are generated from `_build.py`. To edit content, modify
the corresponding `NB_*` cell list in that file and re-run:

```bash
python notebooks/tutorials/_build.py
```

Edit the lists, not the .ipynb files directly — the build script
overwrites them.

## What's not covered

The tutorials focus on the core library. For these specialised paths,
look at:

- **Multi-stream / batched tracking** — `unitrack.tracker.BatchTracker`
  (vmap-batched). See its tests in `tests/unitrack/tracker/test_batch_tracker.py`.
- **Clip-based inference** (MinVIS, DVIS++ patterns) —
  `unitrack.tracker.ClipTracker`. See the spec's §1 amendment.
- **Differentiable tracking** — `Tracker(differentiable=True)` swaps in
  soft companions automatically.
- **Real Optuna sweep on Mask2Former-Cityscapes detections** —
  `examples/hpo_sweep/`.
