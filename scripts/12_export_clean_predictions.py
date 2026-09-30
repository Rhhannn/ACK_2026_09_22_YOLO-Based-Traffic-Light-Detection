import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from common.project_paths import DATASET_ROOT, MODEL_WEIGHTS, OUTPUTS_ROOT

TEST_ROOT = DATASET_ROOT / "images" / "test"
OUTPUT_ROOT = OUTPUTS_ROOT / "temporal_predictions"
MODELS = MODEL_WEIGHTS

CLASS_NAMES = ["red", "yellow", "green", "red_yellow"]

# The primary temporal analysis will use confidence 0.25.
# Exporting from 0.01 preserves low-confidence predictions so that
# confidence sensitivity analyses can be performed without rerunning models.
EXPORT_CONFIDENCE = 0.01
IMAGE_SIZE = 1280
BATCH_SIZE = 4
NMS_IOU = 0.70
MAX_DETECTIONS = 300


def prediction_record(result: object) -> dict:
    result_path = Path(result.path).resolve()
    relative_path = result_path.relative_to(TEST_ROOT.resolve()).as_posix()
    predictions = []

    if result.boxes is not None and len(result.boxes) > 0:
        coordinates = result.boxes.xyxy.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()

        for xyxy, confidence, class_value in zip(
            coordinates,
            confidences,
            classes,
        ):
            class_id = int(class_value)

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise RuntimeError(f"Invalid class ID: {class_id}")

            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": round(float(confidence), 7),
                    "xyxy": [round(float(value), 3) for value in xyxy],
                }
            )

    return {
        "image_path": relative_path,
        "predictions": predictions,
    }


def export_model(
    model_name: str,
    model_path: Path,
    sequence_folders: list[Path],
    expected_image_paths: set[str],
) -> dict:
    final_path = OUTPUT_ROOT / f"{model_name}_test_predictions.jsonl"
    temporary_path = OUTPUT_ROOT / f"{model_name}_test_predictions.jsonl.tmp"

    if temporary_path.exists():
        temporary_path.unlink()

    print()
    print("=" * 80)
    print(f"START: {model_name}")
    print(f"MODEL: {model_path}")
    print(f"SEQUENCE FOLDERS: {len(sequence_folders)}")
    print(f"EXPECTED IMAGES: {len(expected_image_paths)}")
    print(f"EXPORT CONFIDENCE: {EXPORT_CONFIDENCE}")
    print(f"IMAGE SIZE: {IMAGE_SIZE}")
    print(f"BATCH SIZE: {BATCH_SIZE}")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = YOLO(str(model_path))
    processed_paths = set()
    prediction_count = 0
    start_time = time.perf_counter()

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for folder_index, sequence_folder in enumerate(
                sequence_folders,
                start=1,
            ):
                results = model.predict(
                    source=str(sequence_folder),
                    imgsz=IMAGE_SIZE,
                    batch=BATCH_SIZE,
                    device=0,
                    conf=EXPORT_CONFIDENCE,
                    iou=NMS_IOU,
                    max_det=MAX_DETECTIONS,
                    agnostic_nms=False,
                    augment=False,
                    stream=True,
                    verbose=False,
                    save=False,
                )

                for result in results:
                    record = prediction_record(result)
                    relative_path = record["image_path"]

                    if relative_path in processed_paths:
                        raise RuntimeError(
                            f"Duplicate prediction image: {relative_path}"
                        )

                    processed_paths.add(relative_path)
                    prediction_count += len(record["predictions"])
                    output_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                output_file.flush()

                if folder_index % 10 == 0 or folder_index == len(
                    sequence_folders
                ):
                    print(
                        f"{model_name}: folders "
                        f"{folder_index}/{len(sequence_folders)}, "
                        f"images {len(processed_paths)}/"
                        f"{len(expected_image_paths)}"
                    )

        missing_paths = expected_image_paths - processed_paths
        unexpected_paths = processed_paths - expected_image_paths

        if missing_paths:
            examples = sorted(missing_paths)[:5]
            raise RuntimeError(
                f"Missing {len(missing_paths)} images. Examples: {examples}"
            )

        if unexpected_paths:
            examples = sorted(unexpected_paths)[:5]
            raise RuntimeError(
                f"Unexpected {len(unexpected_paths)} images. Examples: {examples}"
            )

        os.replace(temporary_path, final_path)

    finally:
        del model
        torch.cuda.empty_cache()

    elapsed_minutes = (time.perf_counter() - start_time) / 60
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    summary = {
        "model": model_name,
        "image_count": len(processed_paths),
        "prediction_count": prediction_count,
        "elapsed_minutes": round(elapsed_minutes, 3),
        "peak_gpu_memory_gb": round(peak_memory_gb, 3),
        "output_path": str(final_path),
        "settings": {
            "confidence": EXPORT_CONFIDENCE,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "nms_iou": NMS_IOU,
            "max_detections": MAX_DETECTIONS,
            "augment": False,
        },
    }

    print()
    print(f"FINISHED: {model_name}")
    print(f"IMAGES: {summary['image_count']}")
    print(f"PREDICTIONS: {summary['prediction_count']}")
    print(f"MINUTES: {summary['elapsed_minutes']:.1f}")
    print(f"PEAK GPU MEMORY: {summary['peak_gpu_memory_gb']:.2f} GB")
    print(f"SAVED: {final_path}")

    return summary


def main() -> None:
    print("=" * 80)
    print("TEMPORAL PREDICTION EXPORT - VERIFIED VERSION")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    if not TEST_ROOT.exists():
        raise FileNotFoundError(f"Test image root not found: {TEST_ROOT}")

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found for {model_name}: {model_path}"
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(TEST_ROOT.rglob("*.png"))

    if not image_paths:
        raise RuntimeError("No test PNG images were found.")

    expected_image_paths = {
        image_path.resolve().relative_to(TEST_ROOT.resolve()).as_posix()
        for image_path in image_paths
    }

    sequence_folders = sorted(
        {image_path.parent for image_path in image_paths},
        key=lambda path: str(path),
    )

    print(f"TEST IMAGES: {len(expected_image_paths)}")
    print(f"SEQUENCE FOLDERS: {len(sequence_folders)}")
    print(f"MODELS: {', '.join(MODELS.keys())}")
    print(f"EXPORT CONFIDENCE: {EXPORT_CONFIDENCE}")
    print(f"BATCH SIZE: {BATCH_SIZE}")

    summaries = []

    for model_name, model_path in MODELS.items():
        summaries.append(
            export_model(
                model_name=model_name,
                model_path=model_path,
                sequence_folders=sequence_folders,
                expected_image_paths=expected_image_paths,
            )
        )

    summary_path = OUTPUT_ROOT / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summaries, summary_file, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("ALL MODELS EXPORTED AND VERIFIED")
    print("=" * 80)

    for summary in summaries:
        print(
            f"{summary['model']}: "
            f"images={summary['image_count']}, "
            f"predictions={summary['prediction_count']}, "
            f"minutes={summary['elapsed_minutes']:.1f}"
        )

    print(f"SUMMARY: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
