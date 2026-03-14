"""Markdown + JSON writers for benchmark results."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .types import BenchmarkResult

_COLUMNS = ["HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1"]


def write_markdown(
    results: list[BenchmarkResult], path: str | Path, *, metadata: dict
) -> None:
    """Write a markdown table of ``results`` with a metadata preamble."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# HOTA benchmark — {metadata.get('dataset', 'unknown')}", ""]
    lines += [f"- **{k}**: {v}" for k, v in metadata.items()]
    lines += [
        "",
        "| Model | Tracker | " + " | ".join(_COLUMNS) + " | frames | sec |",
        "|" + "---|" * (len(_COLUMNS) + 4),
    ]
    for r in results:
        cells = " | ".join(f"{r.metrics.get(c, float('nan')):.4f}" for c in _COLUMNS)
        lines.append(
            f"| {r.model_key} | {r.tracker_key} | {cells} "
            f"| {r.num_frames} | {r.seconds:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_json(
    results: list[BenchmarkResult], path: str | Path, *, metadata: dict
) -> None:
    """Write ``results`` as a JSON document with a ``metadata`` block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "results": [dataclasses.asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
