import json

from unitrack.benchmarks.hota.report import write_json, write_markdown
from unitrack.benchmarks.hota.types import BenchmarkResult


def _results():
    return [
        BenchmarkResult(
            model_key="m2f-tiny",
            tracker_key="maskiou",
            metrics={
                "HOTA": 0.61,
                "DetA": 0.6,
                "AssA": 0.62,
                "LocA": 0.8,
                "MOTA": 0.55,
                "IDF1": 0.7,
            },
            num_sequences=2,
            num_frames=40,
            seconds=12.3,
        )
    ]


_METADATA = {
    "dataset": "cityscapes-dvps",
    "mask_iou_threshold": 0.5,
    "min_score": 0.1,
    "offset": 1000,
    "thing_ids": [11, 12, 13, 14, 15, 16, 17, 18],
    "unitrack_version": "2.0.0",
    "transformers_version": "4.40.0",
}


def test_markdown_has_header_and_row(tmp_path):
    p = tmp_path / "out.md"
    write_markdown(_results(), p, metadata=_METADATA)
    text = p.read_text()
    assert "| Model | Tracker | HOTA |" in text
    assert "| m2f-tiny | maskiou |" in text
    assert "cityscapes-dvps" in text
    # formatted metric + count cells are locked
    assert "0.6100" in text  # HOTA 0.61 -> {:.4f}
    assert "| 40 |" in text  # num_frames cell
    # richer metadata appears in the header
    assert "offset" in text
    assert "thing_ids" in text
    assert "unitrack_version" in text
    assert "transformers_version" in text


def test_json_roundtrips(tmp_path):
    p = tmp_path / "out.json"
    write_json(_results(), p, metadata=_METADATA)
    data = json.loads(p.read_text())
    assert data["metadata"]["dataset"] == "cityscapes-dvps"
    assert data["results"][0]["model_key"] == "m2f-tiny"
    assert data["results"][0]["tracker_key"] == "maskiou"
    assert data["results"][0]["metrics"]["HOTA"] == 0.61
    # richer metadata round-trips through write_json
    assert data["metadata"]["offset"] == 1000
    assert data["metadata"]["thing_ids"] == [11, 12, 13, 14, 15, 16, 17, 18]
    assert data["metadata"]["mask_iou_threshold"] == 0.5
    assert data["metadata"]["unitrack_version"] == "2.0.0"
    assert data["metadata"]["transformers_version"] == "4.40.0"
