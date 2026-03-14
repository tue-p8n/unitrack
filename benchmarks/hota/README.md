# HOTA model benchmark

A modality-agnostic harness that scores open-weights HuggingFace models with
HOTA by pairing each model with the `unitrack` tracker for association and
`evaluators` for the metric. The reference configuration evaluates panoptic
segmentation models on the local `cityscapes-dvps` split.

The harness has three plug points:

- **`ModelAdapter`** — turns a frame into per-instance thing masks
  (`FramePrediction`). The shipped adapter wraps any HuggingFace panoptic model
  (`HFPanopticAdapter`).
- **`DatasetAdapter`** — yields sequences of `(image, gt_panoptic)`. The shipped
  adapter reads `cityscapes-dvps` from a local LMDB (`CityscapesDVPSDataset`).
- **`TrackerFactory`** — a `unitrack` mask-IoU tracker (`build_mask_tracker`).

Per frame the runner converts the model output to `unitrack.Detections`, steps
the tracker, recovers a stable track id per detection, renders a predicted
panoptic map (`semantic * offset + instance`), and streams `(gt, pred)` into
`evaluators.presets.mot.panoptic_tracking()`.

## Install

The PyPI benchmark dependencies live in an optional extra; the `evaluators`
package (the HOTA/CLEAR/Identity metric) is not distributed as part of this
project, so it is installed separately:

```bash
uv sync --extra benchmark                 # transformers + huggingface-hub, lmdb + Pillow
uv pip install -e /path/to/evaluators     # the HOTA metric implementation
```

`evaluators` is deliberately kept out of the extra: a path dependency would break
CI, which doesn't have it available. The harness imports all of these lazily and
the tests `importorskip` them, so the core `unitrack` import surface depends on
none of them and the benchmark tests skip cleanly when a dep is absent.

## Run

The CLI is exposed as a module:

```bash
uv run python -m unitrack.benchmarks.hota \
  --dataset cityscapes-dvps \
  --models mask2former-tiny,mask2former-small \
  --device auto \
  --limit-seqs 5 \
  --max-frames 30 \
  --out benchmarks/hota/results
```

Flags:

- `--models` — comma-separated registry keys (see `MODEL_REGISTRY`):
  `mask2former-tiny`, `mask2former-small`, `mask2former-base`,
  `oneformer-large`.
- `--lmdb` — path to the dataset LMDB. Defaults to
  `~/Datasets/cityscapes-dvps/cityscapes-dvps.val.lmdb`.
- `--device` — `auto` (CUDA if available, else CPU), `cpu`, or `cuda`.
- `--limit-seqs` / `--max-frames` — cap the sweep for a quick smoke run.
- `--mask-iou-threshold` — minimum mask IoU for a match (higher = stricter;
  default `0.5`). Internally converted to the `1 - IoU` association cost.
- `--min-score` — drop detections below this confidence.

Results are written to `--out` as
`<dataset>_panoptic.md` (a markdown table) and `<dataset>_panoptic.json` (the
machine-readable record, including the metadata block). Committed sample tables
live under `benchmarks/hota/results/`.

## Tests

The CI-safe tests run without network or GPU and use in-memory fakes:

```bash
uv run pytest tests/unitrack/benchmarks/hota -q
```

Tests that need `evaluators`, `transformers`, or `lmdb` guard their imports with
`pytest.importorskip(...)` and are skipped when the `benchmark` extra is absent.

Two tests are opt-in:

- The live HuggingFace model test downloads weights and runs only when
  `UNITRACK_BENCHMARK_LIVE=1` is set.
- The real-LMDB smoke test runs only if
  `~/Datasets/cityscapes-dvps/cityscapes-dvps.val.lmdb` exists locally.

## Box-MOT extension

The harness is modality-agnostic: HOTA does not care whether instances are masks
or boxes, only that predicted and ground-truth tracks can be matched. Extending
to classic box-tracking benchmarks (e.g. MOTChallenge) requires three new
adapters and a metric swap — no changes to the runner:

- **`DatasetAdapter` (MOTChallenge)** — yield each sequence's frames as
  `(image, gt)`, where `gt` carries per-frame boxes and stable track ids instead
  of a panoptic map. Expose the same `key` / `thing_ids` / `offset` surface the
  runner reads.
- **`ModelAdapter` (detector)** — wrap a box detector (e.g. a HuggingFace
  `AutoModelForObjectDetection`) so each frame returns boxes, class ids, and
  scores. Carry boxes on the prediction in place of masks.
- **`TrackerFactory` (box-IoU)** — build a `unitrack` tracker whose cost is
  box-IoU rather than `MaskIoU`, keeping the same `NoLifecycle` /
  per-detection-id-recovery contract so association ids stay reconstructable.

Then swap the metric: replace `evaluators.presets.mot.panoptic_tracking()` with
`evaluators.presets.mot.mot_challenge(...)`, and stream the per-frame box tracks
into it instead of rendering a panoptic map. The runner's loop (model → tracker
→ id recovery → metric update) is unchanged; only the data carried between the
plug points and the metric preset differ.

## Regression baseline

`results/cityscapes-dvps_panoptic.{md,json}` is a committed baseline for
regression testing. The harness is deterministic (HF models in eval mode + the
Jonker LAP solver + fixed input ⇒ bit-identical metrics), so re-running the
exact slice and diffing the JSON catches regressions in the tracker, the metric,
the model loading, or a dependency bump. Reproduce with:

```bash
uv sync --extra benchmark && uv pip install -e /path/to/evaluators   # see Install above
uv run python -m unitrack.benchmarks.hota \
    --dataset cityscapes-dvps \
    --models mask2former-tiny,mask2former-small,mask2former-base,mask2former-large \
    --device cpu --limit-seqs 5 --out benchmarks/hota/results
```

The baseline is a bounded **5-clip / 30-frame** slice (cheap to re-run); the
recorded `unitrack_version` / `transformers_version` in the JSON metadata pin
the environment a diff should be read against. The full val split is the heavy
sweep for a GPU box (oneformer-large and the full 300-frame set are
CPU-prohibitive). Note the modest scores and negative MOTA: an image-trained
panoptic model paired with a simple mask-IoU tracker over crowded Cityscapes
scenes accrues more false positives + misses than ground-truth objects at the
0.5 IoU threshold — HOTA (the localization/association-balanced metric) is the
headline number, and it scales sensibly with backbone size.
