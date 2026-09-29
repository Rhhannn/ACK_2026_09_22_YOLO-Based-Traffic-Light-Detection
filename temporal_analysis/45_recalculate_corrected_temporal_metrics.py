import csv
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from common.project_paths import OUTPUTS_ROOT, REPOSITORY_ROOT

PROJECT_ROOT = REPOSITORY_ROOT
BASE_SCRIPT = PROJECT_ROOT / "temporal_analysis" / "25_analyze_temporal_metrics.py"

CLEAN_PREDICTION_ROOT = OUTPUTS_ROOT / "temporal_predictions"
DEGRADED_PREDICTION_ROOT = (
    OUTPUTS_ROOT / "degraded_temporal_predictions"
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "DTLD_CORRECTED_TEMPORAL_OUTPUT_ROOT",
        str(OUTPUTS_ROOT / "corrected_temporal_analysis"),
    )
)

MODELS = ("yolov8n", "yolo11n", "yolo12n")
CONDITIONS = (
    "clean",
    "fog_l1",
    "fog_l2",
    "fog_l3",
    "lowlight_l1",
    "lowlight_l2",
    "lowlight_l3",
    "motionblur_l1",
    "motionblur_l2",
    "motionblur_l3",
)

CONFIDENCE = 0.25
MATCH_IOU = 0.50
STABILITY_WINDOW_SECONDS = 2.0
STABILITY_SUCCESS_RATE = 0.80
MIN_STABILITY_OBSERVATIONS = 3
MAX_GAP_SECONDS = 2.0
AUC_HORIZON_SECONDS = 5.0
REPORT_TIMES = (0.0, 1.0, 2.0, 3.0, 5.0)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "corrected_temporal_base",
        BASE_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prediction_path(model, condition):
    filename = f"{model}_test_predictions.jsonl"
    if condition == "clean":
        return CLEAN_PREDICTION_ROOT / filename
    return DEGRADED_PREDICTION_ROOT / condition / filename


def corrected_valid_window(observations, start_index, window_seconds):
    """Return a window only when one continuous segment covers its full span."""
    start_time = observations[start_index]["timestamp"]
    required_end_time = start_time + window_seconds

    segment_end_index = start_index
    for index in range(start_index + 1, len(observations)):
        previous_time = observations[index - 1]["timestamp"]
        current_time = observations[index]["timestamp"]
        if current_time - previous_time > MAX_GAP_SECONDS:
            break
        segment_end_index = index

    # The current continuous segment itself must reach the end of the window.
    if observations[segment_end_index]["timestamp"] < required_end_time:
        return None

    window = []
    for observation in observations[start_index : segment_end_index + 1]:
        if observation["timestamp"] > required_end_time:
            break
        window.append(observation)

    if len(window) < MIN_STABILITY_OBSERVATIONS:
        return None

    return window


def find_stability(base, observations):
    ordered = sorted(observations, key=lambda item: item["timestamp"])
    eligible = False

    for start_index in range(len(ordered)):
        window = corrected_valid_window(
            ordered,
            start_index,
            STABILITY_WINDOW_SECONDS,
        )
        if window is None:
            continue

        eligible = True
        success_rate = sum(item["correct"] for item in window) / len(window)
        if success_rate >= STABILITY_SUCCESS_RATE:
            return {
                "eligible": 1,
                "stable": 1,
                "stable_start_seconds": float(
                    window[0]["timestamp"] - ordered[0]["timestamp"]
                ),
            }

    return {
        "eligible": int(eligible),
        "stable": 0,
        "stable_start_seconds": None,
    }


def max_iou_for_class(base, gt_box, predictions, class_name):
    return max(
        (
            base.iou(gt_box, prediction["xyxy"])
            for prediction in predictions
            if prediction["class_name"] == class_name
        ),
        default=0.0,
    )


def longest_complete_red_miss(observations):
    """Non-Red states and gaps above two seconds both end an episode."""
    ordered = sorted(observations, key=lambda item: item["timestamp"])
    current = []
    best_frames = 0
    best_seconds = 0.0
    previous_timestamp = None

    def finish_episode():
        nonlocal current, best_frames, best_seconds
        if not current:
            return
        frame_count = len(current)
        duration = (
            float(current[-1]["timestamp"] - current[0]["timestamp"])
            if frame_count >= 2
            else 0.0
        )
        best_frames = max(best_frames, frame_count)
        best_seconds = max(best_seconds, duration)
        current = []

    for observation in ordered:
        timestamp = observation["timestamp"]

        if (
            previous_timestamp is not None
            and timestamp - previous_timestamp > MAX_GAP_SECONDS
        ):
            finish_episode()

        if observation["ground_truth_state"] != "red":
            finish_episode()
        elif observation["complete_red_miss"]:
            current.append(observation)
        else:
            finish_episode()

        previous_timestamp = timestamp

    finish_episode()
    return best_frames, best_seconds


def analyze_red_tracks(base, frames, predictions):
    track_observations = defaultdict(list)

    for image_path, frame in frames.items():
        image_predictions = [
            prediction
            for prediction in predictions[image_path]
            if float(prediction["confidence"]) >= CONFIDENCE
        ]

        for ground_truth in frame["objects"]:
            state = ground_truth["state"]
            red_iou = None
            green_iou = None
            complete_red_miss = False
            red_to_green = False

            if state == "red":
                red_iou = max_iou_for_class(
                    base,
                    ground_truth["xyxy"],
                    image_predictions,
                    "red",
                )
                green_iou = max_iou_for_class(
                    base,
                    ground_truth["xyxy"],
                    image_predictions,
                    "green",
                )
                complete_red_miss = red_iou == 0.0
                red_to_green = green_iou >= MATCH_IOU

            track_observations[ground_truth["track_key"]].append(
                {
                    "timestamp": frame["timestamp"],
                    "ground_truth_state": state,
                    "complete_red_miss": complete_red_miss,
                    "red_to_green": red_to_green,
                    "red_iou": red_iou,
                }
            )

    rows = []
    for track_key, observations in track_observations.items():
        red_observations = [
            item
            for item in observations
            if item["ground_truth_state"] == "red"
        ]
        if not red_observations:
            continue

        miss_count = sum(item["complete_red_miss"] for item in red_observations)
        red_to_green_count = sum(
            item["red_to_green"] for item in red_observations
        )
        longest_frames, longest_seconds = longest_complete_red_miss(
            observations
        )

        rows.append(
            {
                "track_key": track_key,
                "red_observations": len(red_observations),
                "complete_miss_count": miss_count,
                "complete_miss_rate": miss_count / len(red_observations),
                "red_to_green_count": red_to_green_count,
                "red_to_green_rate": (
                    red_to_green_count / len(red_observations)
                ),
                "longest_complete_miss_frames": longest_frames,
                "longest_complete_miss_seconds": longest_seconds,
            }
        )

    return rows


def auc5_from_starts(starts):
    contributions = []
    for start in starts:
        if start is None or not math.isfinite(start) or start > AUC_HORIZON_SECONDS:
            contributions.append(0.0)
        else:
            contributions.append(
                (AUC_HORIZON_SECONDS - start) / AUC_HORIZON_SECONDS
            )
    return statistics.mean(contributions) * 100.0 if contributions else 0.0


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def make_audit_rows(new_summary):
    old_primary_path = (
        OUTPUTS_ROOT
        / "degraded_temporal_analysis"
        / "primary_temporal_summary.csv"
    )
    old_auc_path = (
        OUTPUTS_ROOT
        / "cumulative_stability_analysis"
        / "cumulative_stability_summary.csv"
    )
    # The comparison is only meaningful when both legacy result tables exist.
    # Do not manufacture zero-valued baselines in a clean reproduction checkout.
    if not old_primary_path.exists() or not old_auc_path.exists():
        return []

    old_primary = {
        (row["model"], row["condition"]): row
        for row in read_csv(old_primary_path)
    }
    old_auc = {
        (row["model"], row["condition"]): row
        for row in read_csv(old_auc_path)
    }

    audit_rows = []
    for row in new_summary:
        key = (row["model"], row["condition"])
        old_p = old_primary.get(key, {})
        old_a = old_auc.get(key, {})
        old_stable = float(old_p.get("stable_rate", 0.0)) * 100.0
        old_duration = float(
            old_p.get("mean_track_longest_no_overlap_seconds", 0.0)
        )
        old_auc5 = float(old_a.get("auc5_normalized", 0.0)) * 100.0
        audit_rows.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "old_stable_rate_percent": old_stable,
                "corrected_stable_rate_percent": row[
                    "stable_rate_percent"
                ],
                "stable_rate_change_pp": (
                    row["stable_rate_percent"] - old_stable
                ),
                "old_auc5": old_auc5,
                "corrected_auc5": row["auc5"],
                "auc5_change_points": row["auc5"] - old_auc5,
                "old_mean_longest_red_miss_seconds": old_duration,
                "corrected_mean_longest_red_miss_seconds": row[
                    "mean_longest_complete_red_miss_seconds"
                ],
                "duration_change_seconds": (
                    row["mean_longest_complete_red_miss_seconds"]
                    - old_duration
                ),
            }
        )
    return audit_rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    base.MATCH_IOU = MATCH_IOU
    base.valid_window = corrected_valid_window

    reference_predictions = base.load_predictions(
        prediction_path("yolov8n", "clean")
    )
    frames, missing_track_ids, invalid_boxes = base.load_ground_truth(
        set(reference_predictions)
    )

    print("=" * 118)
    print("CORRECTED TEMPORAL METRIC RECALCULATION")
    print("Existing predictions, trained weights, and previous results are read only.")
    print(f"Test images: {len(frames):,}")
    print(f"Tasks: {len(MODELS) * len(CONDITIONS)}")
    print("=" * 118)

    stability_rows = []
    red_rows = []
    summaries = []
    task_number = 0

    for condition in CONDITIONS:
        for model in MODELS:
            task_number += 1
            predictions = base.load_predictions(prediction_path(model, condition))
            if set(predictions) != set(frames):
                raise RuntimeError(
                    f"Prediction image set mismatch: {model} / {condition}"
                )

            tracks = base.build_tracks(frames, predictions, CONFIDENCE)
            combination_stability = []

            for track_key, observations in tracks.items():
                result = find_stability(base, observations)
                row = {
                    "model": model,
                    "condition": condition,
                    "track_key": track_key,
                    **result,
                }
                stability_rows.append(row)
                combination_stability.append(row)

            combination_red = analyze_red_tracks(base, frames, predictions)
            for row in combination_red:
                row["model"] = model
                row["condition"] = condition
                red_rows.append(row)

            eligible = [row for row in combination_stability if row["eligible"]]
            stable = [row for row in eligible if row["stable"]]
            mean_duration = statistics.mean(
                row["longest_complete_miss_seconds"]
                for row in combination_red
            )

            summaries.append(
                {
                    "model": model,
                    "condition": condition,
                    "total_tracks": len(combination_stability),
                    "eligible_tracks": len(eligible),
                    "stable_tracks": len(stable),
                    "stable_rate_percent": 100.0 * len(stable) / len(eligible),
                    "auc5": None,
                    "cumulative_rate_t0_percent": None,
                    "cumulative_rate_t1_percent": None,
                    "cumulative_rate_t2_percent": None,
                    "cumulative_rate_t3_percent": None,
                    "cumulative_rate_t5_percent": None,
                    "red_tracks": len(combination_red),
                    "mean_longest_complete_red_miss_seconds": mean_duration,
                }
            )

            print(
                f"[{task_number:02d}/30] {model:<8} {condition:<15} | "
                f"eligible={len(eligible):>3} | stable={len(stable):>3} | "
                f"Red longest mean={mean_duration:.3f}s"
            )

    stability_index = {
        (row["condition"], row["model"], row["track_key"]): row
        for row in stability_rows
    }

    for condition in CONDITIONS:
        eligible_sets = []
        for model in MODELS:
            eligible_sets.append(
                {
                    row["track_key"]
                    for row in stability_rows
                    if row["condition"] == condition
                    and row["model"] == model
                    and row["eligible"] == 1
                }
            )
        common_keys = sorted(set.intersection(*eligible_sets))

        for model in MODELS:
            starts = []
            for track_key in common_keys:
                row = stability_index[(condition, model, track_key)]
                starts.append(
                    row["stable_start_seconds"] if row["stable"] else None
                )

            target = next(
                row
                for row in summaries
                if row["condition"] == condition and row["model"] == model
            )
            target["eligible_tracks"] = len(common_keys)
            target["stable_tracks"] = sum(start is not None for start in starts)
            target["stable_rate_percent"] = (
                100.0 * target["stable_tracks"] / len(common_keys)
            )
            target["auc5"] = auc5_from_starts(starts)

            for report_time in REPORT_TIMES:
                rate = 100.0 * sum(
                    start is not None and start <= report_time
                    for start in starts
                ) / len(common_keys)
                key = f"cumulative_rate_t{int(report_time)}_percent"
                target[key] = rate

    summary_path = OUTPUT_ROOT / "primary_temporal_summary_corrected.csv"
    stability_path = OUTPUT_ROOT / "stability_track_metrics_corrected.csv"
    red_path = OUTPUT_ROOT / "red_track_metrics_corrected.csv"
    audit_path = OUTPUT_ROOT / "old_vs_corrected_audit.csv"
    json_path = OUTPUT_ROOT / "corrected_temporal_analysis.json"

    audit_rows = make_audit_rows(summaries)
    write_csv(summary_path, summaries)
    write_csv(stability_path, stability_rows)
    write_csv(red_path, red_rows)
    write_csv(audit_path, audit_rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "settings": {
                    "confidence": CONFIDENCE,
                    "match_iou": MATCH_IOU,
                    "stability_window_seconds": STABILITY_WINDOW_SECONDS,
                    "stability_success_rate": STABILITY_SUCCESS_RATE,
                    "minimum_stability_observations": MIN_STABILITY_OBSERVATIONS,
                    "maximum_gap_seconds": MAX_GAP_SECONDS,
                    "auc_horizon_seconds": AUC_HORIZON_SECONDS,
                },
                "corrections": [
                    "A stability window must be fully covered by one continuous track segment.",
                    "Any non-Red ground-truth state ends a Red complete-miss episode.",
                ],
                "missing_track_ids": missing_track_ids,
                "invalid_ground_truth_boxes": invalid_boxes,
                "summary": summaries,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 118)
    print("CORRECTED RECALCULATION COMPLETED")
    print(f"Summary: {summary_path}")
    print(f"Track stability: {stability_path}")
    print(f"Red tracks: {red_path}")
    print(f"Old-vs-corrected audit: {audit_path}")
    print(f"JSON: {json_path}")
    print("=" * 118)


if __name__ == "__main__":
    main()
