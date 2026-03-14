"""Dataset adapters. The first is cityscapes-dvps from a local LMDB."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .types import SequenceSample

_THING_IDS = [11, 12, 13, 14, 15, 16, 17, 18]
_OFFSET = 1000


def _decode_png(buf: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(buf)))


class CityscapesDVPSDataset:
    """Reads ``cityscapes-dvps.{split}.lmdb`` (keys ``seq/frame/{image,panoptic}``)."""

    key = "cityscapes-dvps"
    thing_ids = _THING_IDS
    offset = _OFFSET

    def __init__(
        self,
        lmdb_path: str | Path,
        *,
        limit_seqs: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.lmdb_path = Path(lmdb_path)
        self.limit_seqs = limit_seqs
        self.max_frames = max_frames

    def _index(self, txn) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for raw_key, _ in txn.cursor():
            key = raw_key.decode()
            parts = key.split("/")
            if len(parts) != 3:  # skip metadata keys such as ``__info__``
                continue
            seq, frame, modality = parts
            if modality != "image":
                continue
            index.setdefault(seq, []).append(frame)
        for frames in index.values():
            frames.sort()
        return index

    def sequences(self) -> Iterator[SequenceSample]:
        """Yield one :class:`SequenceSample` per sequence in id order."""
        import lmdb

        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            index = self._index(txn)
        seq_ids = sorted(index)
        if self.limit_seqs is not None:
            seq_ids = seq_ids[: self.limit_seqs]
        for seq in seq_ids:
            frames = index[seq]
            if self.max_frames is not None:
                frames = frames[: self.max_frames]
            yield SequenceSample(
                sequence_id=seq,
                length=len(frames),
                frames=self._iter_frames(env, seq, frames),
            )

    def _iter_frames(self, env, seq, frames) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for frame in frames:
            with env.begin() as txn:
                img_buf = self._require(txn, seq, frame, "image")
                pan_buf = self._require(txn, seq, frame, "panoptic")
            img = _decode_png(img_buf)
            pan = _decode_png(pan_buf)
            yield np.ascontiguousarray(img), pan.astype(np.int64)

    @staticmethod
    def _require(txn, seq: str, frame: str, modality: str) -> bytes:
        key = f"{seq}/{frame}/{modality}"
        buf = txn.get(key.encode())
        if buf is None:
            msg = f"missing LMDB key {key!r}"
            raise KeyError(msg)
        return buf


DATASET_REGISTRY: dict[str, type] = {CityscapesDVPSDataset.key: CityscapesDVPSDataset}
