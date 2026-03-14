import io
from pathlib import Path

import numpy as np
import pytest

lmdb = pytest.importorskip("lmdb")
PIL = pytest.importorskip("PIL.Image")
from PIL import Image  # noqa: E402
from unitrack.benchmarks.hota.datasets import CityscapesDVPSDataset  # noqa: E402


def _png_bytes(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _make_tiny_lmdb(path):
    env = lmdb.open(str(path), subdir=False, map_size=1 << 24)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    pan = np.zeros((4, 6), dtype=np.uint16)
    pan[0:2, :] = 11 * 1000 + 1
    with env.begin(write=True) as txn:
        for seq in (0, 1):
            for frame in (0, 1):
                p = f"{seq:06d}/{frame:06d}"
                txn.put(f"{p}/image".encode(), _png_bytes(rgb, "RGB"))
                txn.put(f"{p}/panoptic".encode(), _png_bytes(pan, "I;16"))
    env.close()


def test_iterates_sequences_and_decodes(tmp_path):
    db = tmp_path / "tiny.lmdb"
    _make_tiny_lmdb(db)
    ds = CityscapesDVPSDataset(db)
    seqs = list(ds.sequences())
    assert [s.sequence_id for s in seqs] == ["000000", "000001"]
    image, gt = next(seqs[0].frames)
    assert image.shape == (4, 6, 3)
    assert image.dtype == np.uint8
    assert gt.shape == (4, 6)
    assert int(gt[0, 0]) == 11 * 1000 + 1
    assert ds.thing_ids == [11, 12, 13, 14, 15, 16, 17, 18]
    assert ds.offset == 1000


def test_limits(tmp_path):
    db = tmp_path / "tiny.lmdb"
    _make_tiny_lmdb(db)
    ds = CityscapesDVPSDataset(db, limit_seqs=1, max_frames=1)
    seqs = list(ds.sequences())
    assert len(seqs) == 1
    assert len(list(seqs[0].frames)) == 1


def test_missing_panoptic_key_raises_keyerror(tmp_path):
    # Write an image key but no matching panoptic key for one frame.
    db = tmp_path / "gappy.lmdb"
    env = lmdb.open(str(db), subdir=False, map_size=1 << 24)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    with env.begin(write=True) as txn:
        txn.put(b"000000/000000/image", _png_bytes(rgb, "RGB"))
    env.close()
    ds = CityscapesDVPSDataset(db)
    (seq,) = list(ds.sequences())
    with pytest.raises(KeyError, match="000000/000000/panoptic"):
        next(seq.frames)


_REAL = Path.home() / "Datasets/cityscapes-dvps/cityscapes-dvps.val.lmdb"


@pytest.mark.skipif(not _REAL.exists(), reason="local cityscapes-dvps not present")
def test_real_lmdb_smoke():
    lmdb = pytest.importorskip("lmdb")  # noqa: F841
    from unitrack.benchmarks.hota.datasets import CityscapesDVPSDataset

    ds = CityscapesDVPSDataset(_REAL, limit_seqs=1, max_frames=2)
    (seq,) = list(ds.sequences())
    image, gt = next(seq.frames)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert gt.ndim == 2
    assert gt.max() >= 1000  # encoded semantic*1000 (+instance)
