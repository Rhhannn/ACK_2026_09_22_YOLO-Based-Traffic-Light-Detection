import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from common.project_paths import OUTPUTS_ROOT, REPOSITORY_ROOT

PROJECT_ROOT = REPOSITORY_ROOT
CORRECTION_SCRIPT = (
    PROJECT_ROOT
    / "temporal_analysis"
    / "45_recalculate_corrected_temporal_metrics.py"
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "DTLD_CORRECTED_SENSITIVITY_OUTPUT_ROOT",
        str(
            OUTPUTS_ROOT
            / "corrected_temporal_analysis"
            / "sensitivity"
        ),
    )
)

MODELS = ("yolov8n", "yolo11n", "yolo12n")
SEVERE_CONDITIONS = ("fog_l3", "lowlight_l3", "motionblur_l3")

PRIMARY_CONFIDENCE = 0.25
PRIMARY_IOU = 0.50
PRIMARY_WINDOW = 2.0
SUCCESS_RATE = 0.80
AUC_HORIZON_SECONDS = 5.0

CONFIDENCE_VALUES = (0.10, 0.25, 0.50)
IOU_VALUES = (0.30, 0.40, 0.50)
WINDOW_VALUES = (1.0, 2.0, 3.0)


def load_correction_module():
    spec = importlib.util.spec_from_file_location(
        "corrected_temporal_functions",
        CORRECTION_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_stability(correction, observations, window_seconds):
    ordered = sorted(observations, key=lambda item: item["timestamp"])
    eligible = False

    for start_index in range(len(ordered)):
        window = correction.corrected_valid_window(
            ordered,
            start_index,
            window_seconds,
        )
        if window is None:
            continue

        eligible = True
        correct_rate = sum(item["correct"] for item in window) / len(window)
        if correct_rate >= SUCCESS_RATE:
            return {
                "eligible": 1,
                "stable": 1,
                "start": float(
                    window[0]["timestamp"] - ordered[0]["timestamp"]
                ),
            }

    return {"eligible": int(eligible), "stable": 0, "start": None}


def auc_contribution(result):
    if result["stable"] != 1 or result["start"] is None:
        return 0.0
    if result["start"] > AUC_HORIZON_SECONDS:
        return 0.0
    return max(
        0.0,
        (AUC_HORIZON_SECONDS - result["start"])
        / AUC_HORIZON_SECONDS,
    )


def setting_definitions():
    settings = []
    for value in CONFIDENCE_VALUES:
        settings.append(
            {
                "family": "confidence",
                "value": value,
                "confidence": value,
                "iou": PRIMARY_IOU,
                "window": PRIMARY_WINDOW,
            }
        )
    for value in IOU_VALUES:
        settings.append(
            {
                "family": "iou",
                "value": value,
                "confidence": PRIMARY_CONFIDENCE,
                "iou": value,
                "window": PRIMARY_WINDOW,
            }
        )
    for value in WINDOW_VALUES:
        settings.append(
            {
                "family": "window",
                "value": value,
                "confidence": PRIMARY_CONFIDENCE,
                "iou": PRIMARY_IOU,
                "window": value,
            }
        )
    return settings


def common_eligible_keys(results, condition, setting, models=MODELS):
    sets = []
    setting_key = (setting["family"], setting["value"])
    for model in models:
        mapping = results[(condition, setting_key, model)]
        sets.append(
            {
                key
                for key, value in mapping.items()
                if value["eligible"] == 1
            }
        )
    return sorted(set.intersection(*sets))


def calculate_auc_sensitivity(correction, base, frames):
    settings = setting_definitions()
    track_results = {}

    task_count = len(SEVERE_CONDITIONS) * len(MODELS)
    task_number = 0

    for condition in SEVERE_CONDITIONS:
        for model in MODELS:
            task_number += 1
            predictions = base.load_predictions(
                correction.prediction_path(model, condition)
            )
            tracks_cache = {}

            for setting in settings:
                match_key = (setting["confidence"], setting["iou"])
                if match_key not in tracks_cache:
                    base.MATCH_IOU = setting["iou"]
                    tracks_cache[match_key] = base.build_tracks(
                        frames,
                        predictions,
                        setting["confidence"],
                    )

                tracks = tracks_cache[match_key]
                mapping = {}
                for track_key, observations in tracks.items():
                    mapping[track_key] = find_stability(
                        correction,
                        observations,
                        setting["window"],
                    )

                setting_key = (setting["family"], setting["value"])
                track_results[(condition, setting_key, model)] = mapping

            print(
                f"AUC5 preparation: {task_number:02d}/{task_count} "
                f"{model:<8} {condition}"
            )

    rows = []
    best_rows = []
    for condition in SEVERE_CONDITIONS:
        for setting in settings:
            keys = common_eligible_keys(
                track_results,
                condition,
                setting,
            )
            model_values = []

            for model in MODELS:
                setting_key = (setting["family"], setting["value"])
                mapping = track_results[(condition, setting_key, model)]
                contributions = [
                    auc_contribution(mapping[key]) for key in keys
                ]
                auc5 = (
                    100.0 * sum(contributions) / len(contributions)
                    if contributions
                    else 0.0
                )
                stable_rate = (
                    100.0
                    * sum(mapping[key]["stable"] for key in keys)
                    / len(keys)
                )
                rows.append(
                    {
                        "condition": condition,
                        "parameter_family": setting["family"],
                        "parameter_value": setting["value"],
                        "confidence": setting["confidence"],
                        "iou": setting["iou"],
                        "window_seconds": setting["window"],
                        "model": model,
                        "eligible_tracks": len(keys),
                        "auc5": auc5,
                        "stable_rate_percent": stable_rate,
                    }
                )
                model_values.append((auc5, model))

            best_auc5, best_model = max(model_values)
            second_auc5 = sorted(model_values, reverse=True)[1][0]
            best_rows.append(
                {
                    "metric": "auc5",
                    "condition": condition,
                    "parameter_family": setting["family"],
                    "parameter_value": setting["value"],
                    "best_model": best_model,
                    "best_value": best_auc5,
                    "margin_over_second": best_auc5 - second_auc5,
                }
            )

    return rows, best_rows


def calculate_red_sensitivity(correction, base, frames):
    rows = []
    best_rows = []

    for condition in SEVERE_CONDITIONS:
        condition_model_values = defaultdict(list)
        for model in MODELS:
            predictions = base.load_predictions(
                correction.prediction_path(model, condition)
            )
            for confidence in CONFIDENCE_VALUES:
                correction.CONFIDENCE = confidence
                red_rows = correction.analyze_red_tracks(
                    base,
                    frames,
                    predictions,
                )
                mean_duration = sum(
                    row["longest_complete_miss_seconds"]
                    for row in red_rows
                ) / len(red_rows)
                rows.append(
                    {
                        "condition": condition,
                        "confidence": confidence,
                        "model": model,
                        "red_tracks": len(red_rows),
                        "mean_longest_complete_miss_seconds": mean_duration,
                    }
                )
                condition_model_values[confidence].append(
                    (mean_duration, model)
                )

        for confidence in CONFIDENCE_VALUES:
            ordered = sorted(condition_model_values[confidence])
            best_value, best_model = ordered[0]
            best_rows.append(
                {
                    "metric": "red_duration",
                    "condition": condition,
                    "parameter_family": "confidence",
                    "parameter_value": confidence,
                    "best_model": best_model,
                    "best_value": best_value,
                    "margin_over_second": ordered[1][0] - best_value,
                }
            )

    correction.CONFIDENCE = PRIMARY_CONFIDENCE
    return rows, best_rows


def make_consistency_summary(best_rows):
    counts = defaultdict(int)
    totals = defaultdict(int)
    for row in best_rows:
        key = (row["metric"], row["parameter_family"])
        counts[(row["metric"], row["parameter_family"], row["best_model"])] += 1
        totals[key] += 1

    rows = []
    for metric, family in sorted(totals):
        for model in MODELS:
            wins = counts[(metric, family, model)]
            rows.append(
                {
                    "metric": metric,
                    "parameter_family": family,
                    "model": model,
                    "settings_won": wins,
                    "total_settings": totals[(metric, family)],
                    "win_rate_percent": (
                        100.0 * wins / totals[(metric, family)]
                    ),
                }
            )
    return rows


def print_results(auc_rows, red_rows, consistency):
    print()
    print("=" * 126)
    print("AUC5 SENSITIVITY IN SEVERE CONDITIONS")
    print("=" * 126)
    print(
        f"{'condition':<16} {'family':<12} {'value':>7} "
        f"{'YOLOv8n':>10} {'YOLO11n':>10} {'YOLO12n':>10} {'best':>10}"
    )
    groups = defaultdict(dict)
    for row in auc_rows:
        key = (
            row["condition"],
            row["parameter_family"],
            row["parameter_value"],
        )
        groups[key][row["model"]] = row["auc5"]

    for key, values in groups.items():
        condition, family, value = key
        best = max(values, key=values.get)
        print(
            f"{condition:<16} {family:<12} {value:>7.2f} "
            f"{values['yolov8n']:>10.2f} {values['yolo11n']:>10.2f} "
            f"{values['yolo12n']:>10.2f} {best:>10}"
        )

    print()
    print("RED COMPLETE-MISS DURATION SENSITIVITY")
    print("Lower is better.")
    print(
        f"{'condition':<16} {'confidence':>10} "
        f"{'YOLOv8n':>10} {'YOLO11n':>10} {'YOLO12n':>10} {'best':>10}"
    )
    red_groups = defaultdict(dict)
    for row in red_rows:
        key = (row["condition"], row["confidence"])
        red_groups[key][row["model"]] = row[
            "mean_longest_complete_miss_seconds"
        ]

    for key, values in red_groups.items():
        condition, confidence = key
        best = min(values, key=values.get)
        print(
            f"{condition:<16} {confidence:>10.2f} "
            f"{values['yolov8n']:>10.3f} {values['yolo11n']:>10.3f} "
            f"{values['yolo12n']:>10.3f} {best:>10}"
        )

    print()
    print("BEST-MODEL CONSISTENCY")
    for row in consistency:
        print(
            f"{row['metric']:<14} {row['parameter_family']:<12} "
            f"{row['model']:<8}: {row['settings_won']}/{row['total_settings']}"
        )


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    correction = load_correction_module()
    base = correction.load_base_module()
    base.valid_window = correction.corrected_valid_window

    reference = base.load_predictions(
        correction.prediction_path("yolov8n", "clean")
    )
    frames, _, _ = base.load_ground_truth(set(reference))

    print("=" * 126)
    print("CORRECTED AUC5 AND RED-DURATION SENSITIVITY ANALYSIS")
    print(f"Severe conditions: {', '.join(SEVERE_CONDITIONS)}")
    print("AUC5: Confidence, IoU, and stability window varied one at a time.")
    print("Red duration: Confidence varied; IoU/window are not applicable.")
    print("=" * 126)

    auc_rows, auc_best = calculate_auc_sensitivity(
        correction,
        base,
        frames,
    )
    red_rows, red_best = calculate_red_sensitivity(
        correction,
        base,
        frames,
    )
    best_rows = auc_best + red_best
    consistency = make_consistency_summary(best_rows)

    auc_path = OUTPUT_ROOT / "corrected_auc5_sensitivity.csv"
    red_path = OUTPUT_ROOT / "corrected_red_duration_sensitivity.csv"
    best_path = OUTPUT_ROOT / "corrected_best_model_by_setting.csv"
    consistency_path = OUTPUT_ROOT / "corrected_best_model_consistency.csv"
    json_path = OUTPUT_ROOT / "corrected_temporal_sensitivity.json"

    write_csv(auc_path, auc_rows)
    write_csv(red_path, red_rows)
    write_csv(best_path, best_rows)
    write_csv(consistency_path, consistency)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "primary_settings": {
                    "confidence": PRIMARY_CONFIDENCE,
                    "iou": PRIMARY_IOU,
                    "window_seconds": PRIMARY_WINDOW,
                },
                "auc5_sensitivity": auc_rows,
                "red_duration_sensitivity": red_rows,
                "best_models": best_rows,
                "consistency": consistency,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_results(auc_rows, red_rows, consistency)
    print()
    print("=" * 126)
    print("CORRECTED SENSITIVITY ANALYSIS COMPLETED")
    print(f"AUC5: {auc_path}")
    print(f"Red duration: {red_path}")
    print(f"Best model: {best_path}")
    print(f"Consistency: {consistency_path}")
    print(f"JSON: {json_path}")
    print("=" * 126)


if __name__ == "__main__":
    main()
