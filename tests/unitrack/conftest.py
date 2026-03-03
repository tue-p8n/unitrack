r"""Common set-up for all tests.

Defines fixtures and ensures testing data is present.
"""

from __future__ import annotations

import typing as T  # noqa: N812
from pathlib import Path

import pytest


@pytest.fixture
def assets_path() -> Path:
    path = Path(__file__).parent.parent / "assets"

    assert path.is_dir()

    return path


class SampleData(T.NamedTuple):
    image_path: Path
    depth_path: Path
    segmentation_path: Path


@pytest.fixture
def sample_data(assets_path: Path) -> list[SampleData]:
    depths_path: Path = assets_path / "depths"
    segmentations_path: Path = assets_path / "segmentations"
    images_path: Path = assets_path / "images"
    ids = [f.stem for f in images_path.iterdir() if f.is_file()]

    return [
        SampleData(
            image_path=(images_path / f"{id_}.png"),
            depth_path=(depths_path / f"{id_}.png"),
            segmentation_path=(segmentations_path / f"{id_}.png"),
        )
        for id_ in ids
    ]
