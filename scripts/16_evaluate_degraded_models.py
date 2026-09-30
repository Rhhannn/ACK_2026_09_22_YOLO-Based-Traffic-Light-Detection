import csv
import gc
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from common.project_paths import DATASET_ROOT, DEGRADED_ROOT, MODEL_WEIGHTS, OUTPUTS_ROOT

CLEAN_DATASET_ROOT = DATASET_ROOT

OUTPUT_ROOT = OUTPUTS_ROOT / "degradation_evaluation"
RUNS_ROOT = OUTPUT_ROOT / "runs"
CHECKPOINT_JSON = OUTPUT_ROOT / "evaluation_checkpoint.json"
ALL_RESULTS_CSV = OUTPUT_ROOT / "all_results.csv"
ROBUSTNESS_CSV = OUTPUT_ROOT / "robustness_summary.csv"
FINAL_JSON = OUTPUT_ROOT / "degradation_evaluation_summary.json"

MODELS = MODEL_WEIGHTS

CONDITIONS = [
    {
        "name": "clean",
        "type": "clean",
        "level": 0,
        "value": None,
        "yaml": CLEAN_DATASET_ROOT / "data.yaml",
    },
    {
        "name": "fog_l1",
        "type": "fog",
        "level": 1,
        "value": 0.15,
        "yaml": DEGRADED_ROOT / "fog_l1" / "data.yaml",
    },
    {
        "name": "fog_l2",
        "type": "fog",
        "level": 2,
        "value": 0.30,
        "yaml": DEGRADED_ROOT / "fog_l2" / "data.yaml",
    },
    {
        "name": "fog_l3",
        "type": "fog",
        "level": 3,
        "value": 0.45,
        "yaml": DEGRADED_ROOT / "fog_l3" / "data.yaml",
    },
    {
        "name": "lowlight_l1",
        "type": "lowlight",
        "level": 1,
        "value": 1.4,
        "yaml": DEGRADED_ROOT / "lowlight_l1" / "data.yaml",
    },
    {
        "name": "lowlight_l2",
        "type": "lowlight",
        "level": 2,
        "value": 2.0,
        "yaml": DEGRADED_ROOT / "lowlight_l2" / "data.yaml",
    },
    {
        "name": "lowlight_l3",
        "type": "lowlight",
        "level": 3,
        "value": 2.8,
        "yaml": DEGRADED_ROOT / "lowlight_l3" / "data.yaml",
    },
    {
        "name": "motionblur_l1",
        "type": "motionblur",
        "level": 1,
        "value": 3,
        "yaml": DEGRADED_ROOT / "motionblur_l1" / "data.yaml",
    },
    {
        "name": "motionblur_l2",
        "type": "motionblur",
        "level": 2,
        "value": 7,
        "yaml": DEGRADED_ROOT / "motionblur_l2" / "data.yaml",
    },
    {
        "name": "motionblur_l3",
        "type": "motionblur",
        "level": 3,
        "value": 11,
        "yaml": DEGRADED_ROOT / "motionblur_l3" / "data.yaml",
    },
]

CLASS_NAMES = ["red", "yellow", "green", "red_yellow"]
IMAGE_SIZE = 1280
BATCH_SIZE = 4
DEVICE = 0
WORKERS = 4


def load_checkpoint():
    if not CHECKPOINT_JSON.exists():
        return []

    with CHECKPOINT_JSON.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return payload.get("evaluations", [])


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".temporary")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    temporary.replace(path)


def save_checkpoint(evaluations):
    save_json(
        CHECKPOINT_JSON,
        {
            "settings": {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
                "workers": WORKERS,
                "split": "test",
                "models": {key: str(value) for key, value in MODELS.items()},
            },
            "evaluations": evaluations,
        },
    )


def result_row(
    model_name,
    condition,
    class_name,
    precision,
    recall,
    map50,
    map50_95,
    inference_ms,
    elapsed_minutes,
):
    return {
        "model": model_name,
        "condition": condition["name"],
        "degradation_type": condition["type"],
        "level": condition["level"],
        "parameter_value": condition["value"],
        "class": class_name,
        "precision": float(precision),
        "recall": float(recall),
        "map50": float(map50),
        "map50_95": float(map50_95),
        "inference_ms_per_image": float(inference_ms),
        "elapsed_minutes": float(elapsed_minutes),
    }


def evaluate_one(model, model_name, condition):
    start_time = time.time()
    run_name = f"{model_name}_{condition['name']}"

    metrics = model.val(
        data=str(condition["yaml"]),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=False,
    )

    elapsed_minutes = (time.time() - start_time) / 60.0
    inference_ms = float(metrics.speed.get("inference", 0.0))
    rows = []

    rows.append(
        result_row(
            model_name=model_name,
            condition=condition,
            class_name="all",
            precision=metrics.box.mp,
            recall=metrics.box.mr,
            map50=metrics.box.map50,
            map50_95=metrics.box.map,
            inference_ms=inference_ms,
            elapsed_minutes=elapsed_minutes,
        )
    )

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_metrics = metrics.box.class_result(class_index)
        rows.append(
            result_row(
                model_name=model_name,
                condition=condition,
                class_name=class_name,
                precision=class_metrics[0],
                recall=class_metrics[1],
                map50=class_metrics[2],
                map50_95=class_metrics[3],
                inference_ms=inference_ms,
                elapsed_minutes=elapsed_minutes,
            )
        )

    return {
        "model": model_name,
        "condition": condition["name"],
        "degradation_type": condition["type"],
        "level": condition["level"],
        "parameter_value": condition["value"],
        "yaml": str(condition["yaml"]),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "inference_ms_per_image": inference_ms,
        "elapsed_minutes": elapsed_minutes,
        "rows": rows,
    }


def flatten_rows(evaluations):
    rows = []
    for evaluation in evaluations:
        rows.extend(evaluation["rows"])
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_robustness_rows(all_rows):
    all_class_rows = [row for row in all_rows if row["class"] == "all"]
    clean_by_model = {
        row["model"]: row
        for row in all_class_rows
        if row["condition"] == "clean"
    }
    output_rows = []

    for row in all_class_rows:
        clean = clean_by_model[row["model"]]
        map50_drop = clean["map50"] - row["map50"]
        map50_95_drop = clean["map50_95"] - row["map50_95"]
        recall_drop = clean["recall"] - row["recall"]

        output_rows.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "degradation_type": row["degradation_type"],
                "level": row["level"],
                "parameter_value": row["parameter_value"],
                "precision": row["precision"],
                "recall": row["recall"],
                "map50": row["map50"],
                "map50_95": row["map50_95"],
                "map50_drop": map50_drop,
                "map50_95_drop": map50_95_drop,
                "recall_drop": recall_drop,
                "map50_retention_percent": (
                    row["map50"] / clean["map50"] * 100.0
                    if clean["map50"] > 0
                    else 0.0
                ),
                "map50_95_retention_percent": (
                    row["map50_95"] / clean["map50_95"] * 100.0
                    if clean["map50_95"] > 0
                    else 0.0
                ),
                "inference_ms_per_image": row["inference_ms_per_image"],
                "elapsed_minutes": row["elapsed_minutes"],
            }
        )

    return output_rows


def print_result(evaluation):
    all_row = next(row for row in evaluation["rows"] if row["class"] == "all")
    print(
        f"RESULT | P={all_row['precision']:.4f} | "
        f"R={all_row['recall']:.4f} | "
        f"mAP50={all_row['map50']:.4f} | "
        f"mAP50-95={all_row['map50_95']:.4f} | "
        f"inference={all_row['inference_ms_per_image']:.2f}ms | "
        f"elapsed={all_row['elapsed_minutes']:.1f}min"
    )


def validate_inputs():
    missing = []

    for model_path in MODELS.values():
        if not model_path.exists():
            missing.append(str(model_path))

    for condition in CONDITIONS:
        if not condition["yaml"].exists():
            missing.append(str(condition["yaml"]))

    if missing:
        lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required files are missing:\n{lines}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available in PyTorch")


def main():
    validate_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    evaluations = load_checkpoint()
    completed = {
        (item["model"], item["condition"])
        for item in evaluations
    }
    total_evaluations = len(MODELS) * len(CONDITIONS)

    print("=" * 116)
    print("CLEAN AND DEGRADED TEST EVALUATION")
    print("=" * 116)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Conditions per model: {len(CONDITIONS)} (Clean + 9 degraded)")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Already completed: {len(completed)}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print("=" * 116)

    evaluation_number = 0

    for model_name, model_path in MODELS.items():
        print()
        print(f"LOADING MODEL: {model_name}")
        print(f"PATH: {model_path}")
        model = YOLO(str(model_path))

        for condition in CONDITIONS:
            evaluation_number += 1
            key = (model_name, condition["name"])

            print()
            print("-" * 116)
            print(
                f"[{evaluation_number}/{total_evaluations}] "
                f"{model_name} | {condition['name']}"
            )

            if key in completed:
                print("SKIP: This evaluation is already saved in the checkpoint.")
                continue

            evaluation = evaluate_one(model, model_name, condition)
            evaluations.append(evaluation)
            completed.add(key)
            save_checkpoint(evaluations)
            print_result(evaluation)

            gc.collect()
            torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    all_rows = flatten_rows(evaluations)
    all_rows.sort(
        key=lambda row: (
            list(MODELS).index(row["model"]),
            [item["name"] for item in CONDITIONS].index(row["condition"]),
            ["all", *CLASS_NAMES].index(row["class"]),
        )
    )

    all_fieldnames = [
        "model",
        "condition",
        "degradation_type",
        "level",
        "parameter_value",
        "class",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "inference_ms_per_image",
        "elapsed_minutes",
    ]
    write_csv(ALL_RESULTS_CSV, all_rows, all_fieldnames)

    robustness_rows = make_robustness_rows(all_rows)
    robustness_fieldnames = [
        "model",
        "condition",
        "degradation_type",
        "level",
        "parameter_value",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "map50_drop",
        "map50_95_drop",
        "recall_drop",
        "map50_retention_percent",
        "map50_95_retention_percent",
        "inference_ms_per_image",
        "elapsed_minutes",
    ]
    write_csv(ROBUSTNESS_CSV, robustness_rows, robustness_fieldnames)

    final_payload = {
        "settings": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "workers": WORKERS,
            "split": "test",
            "gpu": torch.cuda.get_device_name(DEVICE),
        },
        "evaluations": evaluations,
        "robustness_summary": robustness_rows,
    }
    save_json(FINAL_JSON, final_payload)

    print()
    print("=" * 116)
    print("ALL CLEAN AND DEGRADED TEST EVALUATIONS COMPLETED")
    print("=" * 116)
    print(f"Completed evaluations: {len(evaluations)} / {total_evaluations}")
    print(f"All class results: {ALL_RESULTS_CSV}")
    print(f"Robustness summary: {ROBUSTNESS_CSV}")
    print(f"JSON summary: {FINAL_JSON}")
    print("=" * 116)


if __name__ == "__main__":
    main()
