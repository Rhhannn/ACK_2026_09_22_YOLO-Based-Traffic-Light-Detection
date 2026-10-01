"""Portable paths shared by the paper-reproduction scripts.

Every path can be overridden with an environment variable.  The defaults keep
raw/derived data, outputs, and optional released weights inside the repository
tree without exposing the authors' local drive layout.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


RAW_DATA_ROOT = _path_from_env(
    "DTLD_RAW_ROOT",
    REPOSITORY_ROOT / "data" / "raw",
)
LABEL_JSON = _path_from_env(
    "DTLD_LABEL_JSON",
    RAW_DATA_ROOT / "DTLD_Labels_v2.0" / "v2.0" / "DTLD_all.json",
)
DATASET_ROOT = _path_from_env(
    "DTLD_DATASET_ROOT",
    REPOSITORY_ROOT / "data" / "processed" / "DTLD_YOLO_4class",
)
DEGRADED_ROOT = _path_from_env(
    "DTLD_DEGRADED_ROOT",
    REPOSITORY_ROOT / "data" / "processed" / "DTLD_YOLO_4class_degraded",
)
OUTPUTS_ROOT = _path_from_env(
    "DTLD_OUTPUT_ROOT",
    REPOSITORY_ROOT / "outputs",
)
RESULTS_ROOT = REPOSITORY_ROOT / "results"
WEIGHTS_ROOT = _path_from_env(
    "DTLD_WEIGHTS_ROOT",
    REPOSITORY_ROOT / "weights",
)
SPLIT_DIR = REPOSITORY_ROOT / "splits"
DATA_YAML = DATASET_ROOT / "data.yaml"


def _weight_path(model_name: str) -> Path:
    released = WEIGHTS_ROOT / f"{model_name}_best.pt"
    trained = (
        OUTPUTS_ROOT
        / "full_training"
        / f"{model_name}_final"
        / "weights"
        / "best.pt"
    )
    return released if released.exists() else trained


MODEL_WEIGHTS = {
    model_name: _weight_path(model_name)
    for model_name in ("yolov8n", "yolo11n", "yolo12n")
}
