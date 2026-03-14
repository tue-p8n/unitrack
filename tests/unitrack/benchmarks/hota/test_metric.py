import numpy as np
import pytest

pytest.importorskip("evaluators")

from unitrack.benchmarks.hota.metric import PanopticMetricRunner


def test_perfect_tracking_scores_hota_one():
    offset, thing_ids = 1000, [11]
    runner = PanopticMetricRunner(offset=offset, thing_ids=thing_ids)
    gt = np.zeros((8, 8), dtype=np.int64)
    gt[0:4, 0:4] = 11 * offset + 1
    runner.start_sequence(length=3)
    for _ in range(3):
        runner.update(gt_panoptic=gt, pred_panoptic=gt.copy())
    runner.end_sequence()
    out = runner.compute()
    assert out["HOTA"] == pytest.approx(1.0, abs=1e-6)
    assert "DetA" in out
    assert "AssA" in out
