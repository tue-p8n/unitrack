"""CLI entry point: ``python -m unitrack.benchmarks.hota``."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch


@dataclasses.dataclass(frozen=True, slots=True)
class RunConfig:
    """Parsed CLI configuration for one benchmark sweep."""

    dataset: str
    model_keys: list[str]
    tracker_keys: list[str] | None
    lmdb: str | None
    out: str
    device: str
    limit_seqs: int | None
    max_frames: int | None
    mask_iou_threshold: float
    min_score: float

    @property
    def cost_threshold(self) -> float:
        """Internal ``1 - IoU`` association cost from the mask-IoU threshold."""
        return 1.0 - self.mask_iou_threshold


def build_config(argv: list[str]) -> RunConfig:
    """Parse ``argv`` into a :class:`RunConfig`."""
    p = argparse.ArgumentParser(prog="unitrack.benchmarks.hota")
    p.add_argument("--dataset", default="cityscapes-dvps")
    p.add_argument("--models", required=True, help="comma-separated registry keys")
    p.add_argument(
        "--trackers",
        default=None,
        help=(
            "comma-separated tracker registry keys (e.g. "
            "maskiou,cosine,cascade,kalman,learned); when given, sweeps "
            "model x tracker into <dataset>_trackers.{md,json}. Omitting it runs "
            "the legacy mask-IoU-only model sweep into <dataset>_panoptic.{md,json} "
            "(NOT all five trackers)"
        ),
    )
    p.add_argument("--lmdb", default=None, help="path to the dataset LMDB")
    p.add_argument("--out", default="benchmarks/hota/results")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit-seqs", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--mask-iou-threshold",
        type=float,
        default=0.5,
        help="minimum mask IoU for a match",
    )
    p.add_argument("--min-score", type=float, default=0.1)
    a = p.parse_args(argv)
    tracker_keys = (
        None
        if a.trackers is None
        else [k.strip() for k in a.trackers.split(",") if k.strip()]
    )
    return RunConfig(
        dataset=a.dataset,
        model_keys=[k.strip() for k in a.models.split(",") if k.strip()],
        tracker_keys=tracker_keys,
        lmdb=a.lmdb,
        out=a.out,
        device=a.device,
        limit_seqs=a.limit_seqs,
        max_frames=a.max_frames,
        mask_iou_threshold=a.mask_iou_threshold,
        min_score=a.min_score,
    )


def resolve_device(name: str) -> torch.device:
    """Resolve a device name (``auto``/``cpu``/``cuda``) to a ``torch.device``."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _package_version(name: str) -> str:
    """Return an installed package version, or ``"unknown"`` if unavailable."""
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark sweep described by ``argv`` and write the tables."""
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from .datasets import CityscapesDVPSDataset
    from .models import MODEL_REGISTRY, build_model
    from .report import write_json, write_markdown
    from .runner import BenchmarkRunner
    from .tracker import TRACKER_REGISTRY

    cfg = build_config(sys.argv[1:] if argv is None else argv)
    device = resolve_device(cfg.device)
    if cfg.dataset != CityscapesDVPSDataset.key:
        msg = f"unknown dataset {cfg.dataset!r}"
        raise SystemExit(msg)
    # Validate tracker keys before any model/dataset construction so a doomed
    # run fails fast (mirrors the unknown-dataset early exit).
    if cfg.tracker_keys is not None:
        unknown = [k for k in cfg.tracker_keys if k not in TRACKER_REGISTRY]
        if unknown:
            known = ", ".join(sorted(TRACKER_REGISTRY))
            msg = f"unknown tracker key(s) {unknown!r}; known: {known}"
            raise SystemExit(msg)
    default_lmdb = Path.home() / "Datasets/cityscapes-dvps/cityscapes-dvps.val.lmdb"
    dataset = CityscapesDVPSDataset(
        cfg.lmdb or default_lmdb,
        limit_seqs=cfg.limit_seqs,
        max_frames=cfg.max_frames,
    )
    models = [build_model(k, thing_ids=dataset.thing_ids) for k in cfg.model_keys]
    runner = BenchmarkRunner(
        device=device, cost_threshold=cfg.cost_threshold, min_score=cfg.min_score
    )
    metadata = {
        "dataset": cfg.dataset,
        "device": str(device),
        "models": {k: MODEL_REGISTRY[k] for k in cfg.model_keys},
        "limit_seqs": cfg.limit_seqs,
        "max_frames": cfg.max_frames,
        "mask_iou_threshold": cfg.mask_iou_threshold,
        "min_score": cfg.min_score,
        "offset": dataset.offset,
        "thing_ids": list(dataset.thing_ids),
        "unitrack_version": _package_version("unitrack"),
        "transformers_version": _package_version("transformers"),
    }
    out = Path(cfg.out)
    if cfg.tracker_keys is None:
        # Model sweep with the default mask-IoU tracker.
        results = runner.run(models=models, dataset=dataset)
        suffix = "panoptic"
    else:
        # Model x tracker sweep: keys were validated above against the registry.
        trackers = {k: TRACKER_REGISTRY[k] for k in cfg.tracker_keys}
        metadata["trackers"] = list(cfg.tracker_keys)
        results = runner.run(models=models, dataset=dataset, trackers=trackers)
        suffix = "trackers"
    write_markdown(results, out / f"{cfg.dataset}_{suffix}.md", metadata=metadata)
    write_json(results, out / f"{cfg.dataset}_{suffix}.json", metadata=metadata)
    print(f"wrote {out}/{cfg.dataset}_{suffix}.{{md,json}}")  # noqa: T201


if __name__ == "__main__":
    main()
