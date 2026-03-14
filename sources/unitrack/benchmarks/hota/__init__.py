"""
HOTA model-benchmark harness.

Pairs open-weights HuggingFace models with the unitrack tracker and the
evaluators HOTA metric. See ``benchmarks/hota/README.md``.
"""

from .datasets import DATASET_REGISTRY, CityscapesDVPSDataset
from .metric import PanopticMetricRunner
from .models import MODEL_REGISTRY, HFPanopticAdapter, build_model
from .protocols import DatasetAdapter, ModelAdapter, TrackerFactory
from .render import TrackRemap, render_pred_panoptic
from .runner import BenchmarkRunner
from .tracker import build_mask_tracker, default_tracker_factory, ids_per_detection
from .types import BenchmarkResult, FramePrediction, SequenceSample

__all__ = [
    "DATASET_REGISTRY",
    "MODEL_REGISTRY",
    "BenchmarkResult",
    "BenchmarkRunner",
    "CityscapesDVPSDataset",
    "DatasetAdapter",
    "FramePrediction",
    "HFPanopticAdapter",
    "ModelAdapter",
    "PanopticMetricRunner",
    "SequenceSample",
    "TrackRemap",
    "TrackerFactory",
    "build_mask_tracker",
    "build_model",
    "default_tracker_factory",
    "ids_per_detection",
    "render_pred_panoptic",
]
