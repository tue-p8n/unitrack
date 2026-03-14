import pytest
import torch
from unitrack.benchmarks.hota.__main__ import build_config, main, resolve_device


def test_build_config_parses_models_and_limits():
    cfg = build_config(
        [
            "--dataset",
            "cityscapes-dvps",
            "--models",
            "mask2former-tiny,oneformer-large",
            "--limit-seqs",
            "2",
            "--max-frames",
            "5",
            "--device",
            "cpu",
            "--lmdb",
            "/x.lmdb",
            "--out",
            "/tmp/r",
            "--mask-iou-threshold",
            "0.7",
        ]
    )
    assert cfg.model_keys == ["mask2former-tiny", "oneformer-large"]
    assert cfg.limit_seqs == 2
    assert cfg.max_frames == 5
    assert cfg.dataset == "cityscapes-dvps"
    assert cfg.mask_iou_threshold == pytest.approx(0.7)
    # higher mask-IoU threshold -> lower (stricter) association cost
    assert cfg.cost_threshold == pytest.approx(0.3)


def test_resolve_device_cpu():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_build_config_trackers_default_none():
    # Without --trackers, the config records None (model sweep, not tracker sweep).
    cfg = build_config(["--models", "mask2former-tiny"])
    assert cfg.tracker_keys is None


def test_build_config_parses_trackers():
    cfg = build_config(
        ["--models", "mask2former-tiny", "--trackers", "maskiou,cosine, kalman"]
    )
    assert cfg.tracker_keys == ["maskiou", "cosine", "kalman"]


def test_main_unknown_dataset_exits_before_model_load():
    with pytest.raises(SystemExit):
        main(["--dataset", "nope", "--models", "mask2former-tiny"])


def test_main_unknown_tracker_key_exits_before_model_load():
    with pytest.raises(SystemExit):
        main(["--models", "mask2former-tiny", "--trackers", "nope"])
