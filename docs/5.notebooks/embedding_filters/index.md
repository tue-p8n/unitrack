# Embedding / appearance filters

Recursive estimators for the appearance (ReID / DETR query / kernel)
embedding a tracker carries per identity.
A motion Kalman filter (see [Kalman intuition](/notebooks/kalman))
fits embeddings only awkwardly — they are high-dimensional,
live on the unit sphere, and have no motion model —
so trackers use one of the estimators demonstrated here instead.

Each notebook drives a **real `unitrack.Tracker`**,
so it doubles as proof the library supports the method.

| # | Notebook | Method | unitrack API |
|---|---|---|---|
| 1 | [EMA](/notebooks/embedding_filters/ema) | Exponential moving average (steady-state scalar Kalman) | `EMAFuse`, `EMATrack` |
| 2 | [Diagonal Kalman](/notebooks/embedding_filters/kalman_diagonal) | Diagonal random-walk Kalman (adaptive, `O(D)`) | `KalmanLinear`, `KalmanUpdate` |
| 3 | [vMF directional](/notebooks/embedding_filters/vmf_directional) | von Mises-Fisher directional recursive Bayes (on the sphere) | `vmf_state_entries` |
| 4 | [EnKF / information filter](/notebooks/embedding_filters/enkf_information) | Ensemble Kalman (high-`D`) + information filter (exact dual) | `enkf_state_entries`, `information_state_entries` |
| 5 | [Gallery and learned](/notebooks/embedding_filters/gallery_and_learned) | Feature-bank gallery (DeepSORT/MeMOT) + learned propagation (MOTR-style) | `gallery_state_entries`, `GalleryCost`, `LearnedProcess`, `LearnedObservation` |
| 6 | [Summary benchmark](/notebooks/embedding_filters/summary_benchmark) | All six side-by-side: FLOPs, speed vs `D`, accuracy, effort, fail/success | benchmark of all of the above |

## Which one?

- **EMA** — cheap, robust default; no uncertainty.
- **Diagonal Kalman** — EMA cost with adaptive, measurable per-dimension uncertainty.
- **vMF** — the *right* geometry for cosine-normalised embeddings;
  estimate stays on the sphere and `kappa` is a directional confidence.
- **Information filter** — exact Gaussian posterior, additive fusion of many cues (moderate `D`).
- **EnKF** — covariance information when `D` is too large for a dense matrix.
- **Gallery** — memory across appearance change / re-ID gaps.
- **Learned propagation** — the expressive, differentiable, MOTR-style option.

Notebooks are generated from `_build.py` in this directory;
edit the `NB_*` cell lists there, re-run it, then execute the
notebook to refresh outputs. Edit the lists, not the `.ipynb`
files directly — regenerating overwrites them, outputs included.
