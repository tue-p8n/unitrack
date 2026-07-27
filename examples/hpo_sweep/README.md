# Tracker design space HPO with Optuna and Mask2Former-Cityscapes

This example encodes the tracker design space as an Optuna search space,
builds a `unitrack.Tracker` per trial, runs the trial on detections produced
by a Mask2Former model, and scores the result. It is a single-machine,
single-clip miniature — a starting point for your own sweeps.

## What's here

| File | Purpose |
|---|---|
| `search_space.py` | `sample_tracker(trial)` — maps an Optuna trial to a fully-built `unitrack.Tracker` covering every paper-relevant axis (n_stages, fusion_mode, gating, descriptor, cost, association, threshold, kalman, max_age, min_hits). |
| `detector.py` | `Mask2FormerDetector` — wraps a HuggingFace `Mask2FormerForUniversalSegmentation` model and emits per-frame `unitrack.data.Detections` records (kernel embeddings, masks, bboxes, classes, scores, 2D centroids). |
| `objective.py` | `evaluate_config(...)` — runs a tracker on a clip and returns a scalar score. Two modes: a synthetic identity-preservation proxy (default; no model load required) and a real-data path that runs the Mask2Former detector on user-supplied frames. |
| `run_hpo.py` | CLI entry point; runs an Optuna study with a configurable trial budget. |

## Installation

```bash
pip install 'unitrack[hpo]' transformers pillow
```

Or via `uv`:

```bash
uv pip install 'unitrack[hpo]' transformers pillow
```

`transformers` and `pillow` are only needed for the real-data path.

The Mask2Former model itself downloads on first use via HuggingFace Hub.

## Quick start (synthetic data, ~1 minute)

```bash
# 50-trial sweep on a synthetic 8-frame trajectory with 3 ground-truth identities
python -m examples.hpo_sweep.run_hpo \
  --n-trials 50 \
  --n-frames 8 \
  --n-objects 3 \
  --output study.json
```

This requires only `optuna` (no `transformers`). The synthetic objective
generates a clip where each ground-truth identity has a known kernel
embedding plus jittered position; the metric measures how well the tracker
preserves identities (1 − ID-switch rate). It's a *proxy* for HOTA/VPQ
that's adequate for sanity-checking the search space on a single tiny clip.

## Real data path

If you have a directory of frames from a Cityscapes-VPS sequence:

```bash
python -m examples.hpo_sweep.run_hpo \
  --frames-dir path/to/frames/ \
  --model facebook/mask2former-swin-tiny-cityscapes-instance \
  --n-trials 100 \
  --output study.json
```

Replace `--model` with whichever Mask2Former checkpoint you want. The paper
used a ResNet-50 backbone trained on Cityscapes-VPS; a publicly-available
HuggingFace stand-in that exposes the right outputs (per-query class
logits, masks, and decoder kernels) is
`facebook/mask2former-swin-tiny-cityscapes-instance`. For a true
ResNet-50 reproduction, use the original Mask2Former author checkpoints
and adapt `Mask2FormerDetector.from_huggingface(...)` to load them.

The objective in real-data mode uses a *frame-to-frame query-cosine
proxy* — it measures how well consecutive-frame identities, as
established by the tracker, agree with the segmenter's query embeddings.
This is *not* HOTA; for a published evaluation you need a labelled
benchmark and a real metric. See `eval.py` for the proxy details.

## What the search space covers

`search_space.sample_tracker` exposes these knobs to Optuna:

| Parameter | Range | Axis |
|---|---|---|
| `n_stages` | {1, 2, 3, 4} | Number of stages |
| `fusion_mode` | {cascaded, parallel} | Fusion mode |
| `max_age` | [1, 5] frames | Tracklet max age |
| `min_hits` | [1, 3] | Tracklet min hits |
| per-stage `gating` | {none, class, score, spatial, motion} | Gating $G_k$ |
| per-stage `descriptor` | {kernel, mask, bbox} | Descriptor $F_k$ |
| per-stage `cost` | {cosine, cdist, bisoftmax, iou} | Cost $C_k$ |
| per-stage `association` | {jonker, greedy} | Association $A_k$ |
| per-stage `threshold` | [0.05, 0.50] | Cost threshold |
| per-stage `use_kalman` | {true, false} | Use Kalman filter |
| per-branch `weight` (parallel mode) | [0.0, 1.0] | Cost weights |

Depth-aware tracking (paper §4 row "Depth head") is *not* included here —
the example uses the Cityscapes (no-depth) variant of the benchmark. For a
DVPS reproduction, add a `centroid_3d` state with `KalmanCentroid3D` and
include `MotionGate` / `Mahalanobis` cost branches in the search space.

## Reading results

```python
import json, optuna

study = optuna.load_study(study_name="tracker_hpo", storage="sqlite:///study.db")
best = study.best_trial
print(best.params)  # the winning configuration
print(best.value)  # the score (higher is better in synthetic mode)
```

## Caveats

1. **Single clip.** Robust conclusions hinge on aggregation across
   thousands of trials and seven benchmarks. A 50-trial sweep on one clip
   tells you almost nothing about *transferability* — only about which
   configs work on this specific clip.

2. **Proxy metric.** Both objective modes are proxies. The synthetic
   ID-preservation score is reasonable for sanity-checking the search
   space; the query-cosine real-data proxy is much weaker than HOTA/VPQ.
   For publishable numbers, plug in a real evaluator on annotated data.

3. **Detector approximation.** Mask2Former's per-query decoder embeddings
   are exposed via output `class_queries_logits` shape but the actual
   "kernel" used by the paper is the pre-projection query feature. The
   detector here uses the closest-available HuggingFace output
   (post-decoder query embeddings of dim 256). If you control the
   detector training, you may want to expose a more specific embedding.
