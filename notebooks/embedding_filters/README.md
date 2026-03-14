# Embedding / appearance filters

Recursive estimators for the appearance (ReID / DETR query / kernel)
embedding a tracker carries per identity. A motion Kalman filter (see
`../kalman.ipynb`) fits embeddings only awkwardly — they are
high-dimensional, live on the unit sphere, and have no motion model — so
trackers use one of the estimators demonstrated here instead.

Each notebook drives a **real `unitrack.Tracker`**, so it doubles as proof
the library supports the method.

| # | Notebook | Method | unitrack API |
|---|---|---|---|
| 1 | [01_ema.ipynb](01_ema.ipynb) | Exponential moving average (steady-state scalar Kalman) | `EMAFuse`, `EMATrack` |
| 2 | [02_kalman_diagonal.ipynb](02_kalman_diagonal.ipynb) | Diagonal random-walk Kalman (adaptive, `O(D)`) | `KalmanLinear`, `KalmanUpdate` |
| 3 | [03_vmf_directional.ipynb](03_vmf_directional.ipynb) | von Mises-Fisher directional recursive Bayes (on the sphere) | `vmf_state_entries` |
| 4 | [04_enkf_information.ipynb](04_enkf_information.ipynb) | Ensemble Kalman (high-`D`) + information filter (exact dual) | `enkf_state_entries`, `information_state_entries` |
| 5 | [05_gallery_and_learned.ipynb](05_gallery_and_learned.ipynb) | Feature-bank gallery (DeepSORT/MeMOT) + learned propagation (MOTR-style) | `gallery_state_entries`, `GalleryCost`, `LearnedProcess`, `LearnedObservation` |
| 6 | [06_summary_benchmark.ipynb](06_summary_benchmark.ipynb) | All seven side-by-side: FLOPs, speed vs `D`, accuracy, effort, fail/success | benchmark of all of the above |

## Which one?

- **EMA** — cheap, robust default; no uncertainty.
- **Diagonal Kalman** — EMA cost with adaptive, measurable per-dimension uncertainty.
- **vMF** — the *right* geometry for cosine-normalised embeddings; estimate stays on the sphere and `kappa` is a directional confidence.
- **Information filter** — exact Gaussian posterior, additive fusion of many cues (moderate `D`).
- **EnKF** — covariance information when `D` is too large for a dense matrix.
- **Gallery** — memory across appearance change / re-ID gaps.
- **Learned propagation** — the expressive, differentiable, MOTR-style option.

## Running

```bash
uv run jupyter lab notebooks/embedding_filters/
```

The notebooks are generated from `_build.py`; edit the `NB_*` lists there
and re-run `python notebooks/embedding_filters/_build.py`, then execute to
refresh outputs. Edit the lists, not the `.ipynb` files directly.
