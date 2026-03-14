"""State protocols and :class:`~unitrack.states.State` recipes for tracklet fields."""

from __future__ import annotations

from .base import Initializer, Observation, Process, State
from .directional import (
    VonMisesFisherDecay,
    VonMisesFisherUpdate,
    vmf_state_entries,
)
from .ema import EMADecay, EMAFuse, EMATrack, WeightedFuse
from .gallery import GalleryAppend, GalleryInitializer, gallery_state_entries
from .identity import (
    ConstantInitializer,
    EyeInitializer,
    FromDetectionField,
    Identity,
    NoopObservation,
    NoopProcess,
    NormalizedFromDetectionField,
    PadZerosInitializer,
    Replace,
    ScaledFromDetectionField,
    ZerosInitializer,
)
from .kalman import (
    EnsembleInitializer,
    EnsembleProcess,
    EnsembleUpdate,
    InformationProcess,
    InformationUpdate,
    KalmanBBox,
    KalmanCentroid2D,
    KalmanCentroid3D,
    KalmanLinear,
    KalmanUpdate,
    enkf_state_entries,
    information_state_entries,
)
from .learned import LearnedObservation, LearnedProcess
from .soft import SoftReplace

__all__ = [
    "ConstantInitializer",
    "EMADecay",
    "EMAFuse",
    "EMATrack",
    "EnsembleInitializer",
    "EnsembleProcess",
    "EnsembleUpdate",
    "EyeInitializer",
    "FromDetectionField",
    "GalleryAppend",
    "GalleryInitializer",
    "Identity",
    "InformationProcess",
    "InformationUpdate",
    "Initializer",
    "KalmanBBox",
    "KalmanCentroid2D",
    "KalmanCentroid3D",
    "KalmanLinear",
    "KalmanUpdate",
    "LearnedObservation",
    "LearnedProcess",
    "NoopObservation",
    "NoopProcess",
    "NormalizedFromDetectionField",
    "Observation",
    "PadZerosInitializer",
    "Process",
    "Replace",
    "ScaledFromDetectionField",
    "SoftReplace",
    "State",
    "VonMisesFisherDecay",
    "VonMisesFisherUpdate",
    "WeightedFuse",
    "ZerosInitializer",
    "enkf_state_entries",
    "gallery_state_entries",
    "information_state_entries",
    "vmf_state_entries",
]
