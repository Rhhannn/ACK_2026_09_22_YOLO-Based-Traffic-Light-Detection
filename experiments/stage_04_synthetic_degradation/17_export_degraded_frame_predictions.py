import gc
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from src.temporal_robustness.repository_paths import DEGRADED_ROOT, MODEL_WEIGHTS, OUTPUTS_ROOT

OUTPUT_ROOT = OUTPUTS_ROOT / "degraded_temporal_predictions"
SUMMARY_PATH = OUTPUT_ROOT / "export_summary.json"

MODELS = MODEL_WEIGHTS

CONDITIONS = [
    "fog_l1",
    "fog_l2",
    "fog_l3",
    "lowlight_l1",
    "lowlight_l2",
    "lowlight_l3",
    "motionblur_l1",
    "motionblur_l2",
    "motionblur_l3",
]

CLASS_NAMES = ["red", "yellow", "green", "red_yellow"]

# Predictions are stored from a low threshold so that later analyses can test
# confidence 0.10, 0.25, and 0.50 without rerunning inference.
EXPORT_CONFIDENCE = 0.01
IMAGE_SIZE = 1280
BATCH_SIZE = 4
DEVICE = 0
NMS_IOU = 0.70
MAX_DETECTIONS = 300
EXPECTED_IMAGES = 4244


def save_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".temporary")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    os.replace(temporary, path)


def load_summary():
    if not SUMMARY_PATH.exists():
        return {"settings": {}, "tasks": {}}

    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload.get("tasks"), dict):
        payload["tasks"] = {}

    return payload


def prediction_record(result, test_root):
    result_path = Path(result.path).resolve()
    relative_path = result_path.relative_to(test_root.resolve()).as_posix()
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


def validate_prediction_file(path, expected_paths):
    if not path.exists():
        return {
            "valid": False,
            "image_count": 0,
            "prediction_count": 0,
            "reason": "file does not exist",
        }

    image_paths = set()
    prediction_count = 0

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)
                image_path = record["image_path"]

                if image_path in image_paths:
                    return {
                        "valid": False,
                        "image_count": len(image_paths),
                        "prediction_count": prediction_count,
                        "reason": f"duplicate image at line {line_number}",
                    }

                image_paths.add(image_path)
                prediction_count += len(record.get("predictions", []))

    except Exception as error:
        return {
            "valid": False,
            "image_count": len(image_paths),
            "prediction_count": prediction_count,
            "reason": repr(error),
        }

    missing = expected_paths - image_paths
    unexpected = image_paths - expected_paths

    if missing or unexpected:
        return {
            "valid": False,
            "image_count": len(image_paths),
            "prediction_count": prediction_count,
            "reason": (
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            ),
        }

    return {
        "valid": True,
        "image_count": len(image_paths),
        "prediction_count": prediction_count,
        "reason": None,
    }


def export_task(
    model,
    model_name,
    condition,
    test_root,
    sequence_folders,
    expected_paths,
):
    condition_output = OUTPUT_ROOT / condition
    condition_output.mkdir(parents=True, exist_ok=True)
    final_path = condition_output / f"{model_name}_test_predictions.jsonl"
    temporary_path = condition_output / (
        f"{model_name}_test_predictions.jsonl.temporary"
    )

    existing = validate_prediction_file(final_path, expected_paths)

    if existing["valid"]:
        print(
            f"SKIP: verified existing file | images="
            f"{existing['image_count']:,} | predictions="
            f"{existing['prediction_count']:,}"
        )
        return {
            "status": "completed",
            "skipped": True,
            "model": model_name,
            "condition": condition,
            "image_count": existing["image_count"],
            "prediction_count": existing["prediction_count"],
            "elapsed_minutes": None,
            "peak_gpu_memory_gb": None,
            "output_path": str(final_path),
        }

    if final_path.exists():
        print(
            f"Existing output is incomplete or invalid; it will be replaced "
            f"after a successful rerun. Reason: {existing['reason']}"
        )

    if temporary_path.exists():
        temporary_path.unlink()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
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
                    device=DEVICE,
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
                    record = prediction_record(result, test_root)
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

                if (
                    folder_index % 20 == 0
                    or folder_index == len(sequence_folders)
                ):
                    print(
                        f"  folders {folder_index}/{len(sequence_folders)} | "
                        f"images {len(processed_paths):,}/"
                        f"{len(expected_paths):,}"
                    )

        missing = expected_paths - processed_paths
        unexpected = processed_paths - expected_paths

        if missing or unexpected:
            raise RuntimeError(
                f"Prediction path mismatch: missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )

        temporary_validation = validate_prediction_file(
            temporary_path,
            expected_paths,
        )

        if not temporary_validation["valid"]:
            raise RuntimeError(
                "Temporary prediction file validation failed: "
                f"{temporary_validation['reason']}"
            )

        os.replace(temporary_path, final_path)

    except KeyboardInterrupt:
        print()
        print(
            "INTERRUPTED: the current unfinished condition will be rerun next "
            "time; earlier completed conditions remain saved."
        )
        raise

    elapsed_minutes = (time.perf_counter() - start_time) / 60.0
    peak_memory_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)

    return {
        "status": "completed",
        "skipped": False,
        "model": model_name,
        "condition": condition,
        "image_count": len(processed_paths),
        "prediction_count": prediction_count,
        "elapsed_minutes": round(elapsed_minutes, 3),
        "peak_gpu_memory_gb": round(peak_memory_gb, 3),
        "output_path": str(final_path),
    }


def validate_inputs():
    missing = []

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available in PyTorch")

    for model_path in MODELS.values():
        if not model_path.exists():
            missing.append(str(model_path))

    for condition in CONDITIONS:
        test_root = DEGRADED_ROOT / condition / "images" / "test"
        if not test_root.exists():
            missing.append(str(test_root))

    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{joined}")


def main():
    validate_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    summary["settings"] = {
        "export_confidence": EXPORT_CONFIDENCE,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "nms_iou": NMS_IOU,
        "max_detections": MAX_DETECTIONS,
        "expected_images_per_task": EXPECTED_IMAGES,
        "models": {key: str(value) for key, value in MODELS.items()},
        "conditions": CONDITIONS,
        "gpu": torch.cuda.get_device_name(DEVICE),
    }

    total_tasks = len(MODELS) * len(CONDITIONS)
    task_number = 0

    print("=" * 112)
    print("EXPORT DEGRADED TEST PREDICTIONS FOR TEMPORAL ANALYSIS")
    print("=" * 112)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Degraded conditions: {len(CONDITIONS)}")
    print(f"Total tasks: {total_tasks}")
    print(f"Images per task: {EXPECTED_IMAGES:,}")
    print(f"Total inference images: {total_tasks * EXPECTED_IMAGES:,}")
    print(f"Export confidence: {EXPORT_CONFIDENCE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print("=" * 112)

    for model_name, model_path in MODELS.items():
        print()
        print(f"LOADING MODEL: {model_name}")
        print(f"MODEL PATH: {model_path}")
        model = YOLO(str(model_path))

        for condition in CONDITIONS:
            task_number += 1
            test_root = DEGRADED_ROOT / condition / "images" / "test"
            image_paths = sorted(test_root.rglob("*.png"))

            if len(image_paths) != EXPECTED_IMAGES:
                raise RuntimeError(
                    f"{condition}: expected {EXPECTED_IMAGES} images, "
                    f"found {len(image_paths)}"
                )

            expected_paths = {
                path.resolve().relative_to(test_root.resolve()).as_posix()
                for path in image_paths
            }
            sequence_folders = sorted(
                {path.parent for path in image_paths},
                key=lambda path: str(path),
            )
            task_key = f"{model_name}/{condition}"

            print()
            print("-" * 112)
            print(
                f"[{task_number}/{total_tasks}] MODEL={model_name} | "
                f"CONDITION={condition}"
            )
            print(
                f"Sequence folders: {len(sequence_folders)} | "
                f"images: {len(expected_paths):,}"
            )

            task_result = export_task(
                model=model,
                model_name=model_name,
                condition=condition,
                test_root=test_root,
                sequence_folders=sequence_folders,
                expected_paths=expected_paths,
            )
            summary["tasks"][task_key] = task_result
            save_json_atomic(SUMMARY_PATH, summary)

            print(
                f"COMPLETED: images={task_result['image_count']:,} | "
                f"predictions={task_result['prediction_count']:,} | "
                f"minutes={task_result['elapsed_minutes']} | "
                f"peak GPU GB={task_result['peak_gpu_memory_gb']}"
            )

            gc.collect()
            torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    completed_tasks = sum(
        task.get("status") == "completed"
        for task in summary["tasks"].values()
    )

    print()
    print("=" * 112)
    print("DEGRADED TEMPORAL PREDICTION EXPORT COMPLETED")
    print("=" * 112)
    print(f"Completed tasks: {completed_tasks} / {total_tasks}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Summary: {SUMMARY_PATH}")
    print("=" * 112)


if __name__ == "__main__":
    main()
