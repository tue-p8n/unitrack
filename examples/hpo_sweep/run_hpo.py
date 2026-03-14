r"""
CLI entry point for the tracker HPO study.

Runs an Optuna sweep over the tracker design space defined in ``search_space.py``,
on either a synthetic clip (default) or a directory of frames passed through
a Mask2Former detector.

Usage::

    # Synthetic-only (no transformers required)
    python -m examples.hpo_sweep.run_hpo \
        --n-trials 50 --output study.json

    # Real frames + Mask2Former
    python -m examples.hpo_sweep.run_hpo \
        --frames-dir path/to/frames \
        --model facebook/mask2former-swin-tiny-cityscapes-instance \
        --n-trials 100 --output study.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

try:
    import optuna
except ImportError as _err:
    _msg = "optuna is required for the HPO sweep: pip install 'unitrack[hpo]'"
    raise ImportError(_msg) from _err

from .objective import real_data_objective, synthetic_clip, synthetic_objective
from .search_space import TrackerSchema


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tracker design space HPO sweep.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("study.json"),
        help="Where to write the best-trial summary.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///study.db). Default in-memory.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="tracker_hpo",
        help="Optuna study name.",
    )

    # Synthetic-mode knobs.
    parser.add_argument(
        "--n-frames",
        type=int,
        default=8,
        help="Frames per synthetic clip.",
    )
    parser.add_argument(
        "--n-objects",
        type=int,
        default=3,
        help="Ground-truth identities per synthetic clip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the synthetic clip.",
    )

    # Real-mode knobs.
    parser.add_argument(
        "--frames-dir",
        type=pathlib.Path,
        default=None,
        help="If set, run real-data objective over the frame images here.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/mask2former-swin-tiny-cityscapes-instance",
        help="HuggingFace model id for Mask2Former.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the detector (e.g. 'cuda:0').",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Detector confidence threshold.",
    )

    parser.add_argument(
        "--kernel-dim",
        type=int,
        default=256,
        help="Mask2Former decoder hidden dim. Match your model's hidden size.",
    )
    parser.add_argument(
        "--mask-h",
        type=int,
        default=96,
        help="Resampled mask height stored on Tracklets.",
    )
    parser.add_argument(
        "--mask-w",
        type=int,
        default=192,
        help="Resampled mask width stored on Tracklets.",
    )
    return parser


def main() -> None:
    """Run the Optuna sweep per the CLI arguments."""
    args = _build_parser().parse_args()

    schema = TrackerSchema(
        kernel_dim=args.kernel_dim,
        mask_shape=(args.mask_h, args.mask_w),
    )

    if args.frames_dir is None:
        clip = synthetic_clip(
            n_frames=args.n_frames,
            n_objects=args.n_objects,
            schema=schema,
            seed=args.seed,
        )

        def objective(trial: optuna.trial.Trial) -> float:
            return synthetic_objective(trial, clip=clip)

    else:
        from .detector import Mask2FormerDetector  # noqa: PLC0415 — optional dep

        frame_paths = sorted(args.frames_dir.glob("*.png"))
        if not frame_paths:
            frame_paths = sorted(args.frames_dir.glob("*.jpg"))
        if not frame_paths:
            msg = f"no frames found in {args.frames_dir!s}"
            raise FileNotFoundError(msg)

        detector = Mask2FormerDetector.from_huggingface(
            model_name=args.model,
            device=args.device,
            score_threshold=args.score_threshold,
            target_mask_shape=(args.mask_h, args.mask_w),
        )

        def objective(trial: optuna.trial.Trial) -> float:
            return real_data_objective(
                trial,
                detector=detector,
                frame_paths=frame_paths,
                schema=schema,
            )

    if args.storage is None:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
        )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    summary = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
