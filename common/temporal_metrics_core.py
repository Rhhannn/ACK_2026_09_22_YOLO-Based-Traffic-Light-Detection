import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath

from common.project_paths import LABEL_JSON, OUTPUTS_ROOT

JSON_PATH = LABEL_JSON
PREDICTION_ROOT = OUTPUTS_ROOT / "temporal_predictions"
OUTPUT_ROOT = OUTPUTS_ROOT / "temporal_analysis"

PREDICTION_PATHS = {
    "yolov8n": PREDICTION_ROOT / "yolov8n_test_predictions.jsonl",
    "yolo11n": PREDICTION_ROOT / "yolo11n_test_predictions.jsonl",
    "yolo12n": PREDICTION_ROOT / "yolo12n_test_predictions.jsonl",
}

CLASS_NAMES = ["red", "yellow", "green", "red_yellow"]
TARGET_STATES = set(CLASS_NAMES)

CONFIDENCE_THRESHOLDS = [0.10, 0.25, 0.50]
STABILITY_WINDOWS = [1.5, 2.0, 3.0]

PRIMARY_CONFIDENCE = 0.25
PRIMARY_STABILITY_WINDOW = 2.0

MATCH_IOU = 0.50
MIN_STABILITY_OBSERVATIONS = 3
STABILITY_SUCCESS_RATE = 0.80
MAX_CONSECUTIVE_GAP_SECONDS = 2.0


def mean_or_none(values):
    return statistics.mean(values) if values else None


def median_or_none(values):
    return statistics.median(values) if values else None


def rounded(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0,
        box_a[3] - box_a[1],
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0,
        box_b[3] - box_b[1],
    )
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_predictions(path):
    predictions_by_image = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            image_path = record["image_path"]

            if image_path in predictions_by_image:
                raise RuntimeError(
                    f"Duplicate prediction at line {line_number}: {image_path}"
                )

            predictions_by_image[image_path] = record["predictions"]

    return predictions_by_image


def original_to_png_path(original_path):
    normalized = original_path.replace("\\", "/")

    if normalized.startswith("./"):
        normalized = normalized[2:]

    return PurePosixPath(normalized).with_suffix(".png").as_posix()


def load_ground_truth(test_image_paths):
    with JSON_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    frames = {}
    missing_track_ids = 0
    invalid_boxes = 0

    for image in data["images"]:
        converted_path = original_to_png_path(image["image_path"])

        if converted_path not in test_image_paths:
            continue

        path_parts = PurePosixPath(converted_path).parts
        sequence_name = "/".join(path_parts[:3])
        timestamp = float(image["time_stamp"])
        objects = []

        for label_index, label in enumerate(image.get("labels", [])):
            attributes = label.get("attributes", {})

            if attributes.get("relevance") != "relevant":
                continue

            if attributes.get("pictogram") != "circle":
                continue

            state = attributes.get("state")

            if state not in TARGET_STATES:
                continue

            x = float(label["x"])
            y = float(label["y"])
            width = float(label["w"])
            height = float(label["h"])

            if width <= 0 or height <= 0:
                invalid_boxes += 1
                continue

            track_id = label.get("track_id")

            if track_id is None:
                missing_track_ids += 1
                track_id = f"missing_track_{label_index}"

            objects.append(
                {
                    "state": state,
                    "track_id": str(track_id),
                    "track_key": f"{sequence_name}/{track_id}",
                    "xyxy": [x, y, x + width, y + height],
                }
            )

        frames[converted_path] = {
            "image_path": converted_path,
            "sequence": sequence_name,
            "timestamp": timestamp,
            "objects": objects,
        }

    missing_images = test_image_paths - set(frames)

    if missing_images:
        raise RuntimeError(
            f"Ground truth is missing {len(missing_images)} test images. "
            f"Examples: {sorted(missing_images)[:5]}"
        )

    return frames, missing_track_ids, invalid_boxes


def match_frame(frame, predictions, confidence_threshold):
    filtered_predictions = [
        prediction
        for prediction in predictions
        if float(prediction["confidence"]) >= confidence_threshold
    ]

    candidates = []

    for ground_truth_index, ground_truth in enumerate(frame["objects"]):
        for prediction_index, prediction in enumerate(filtered_predictions):
            overlap = iou(ground_truth["xyxy"], prediction["xyxy"])

            if overlap >= MATCH_IOU:
                candidates.append(
                    (
                        overlap,
                        float(prediction["confidence"]),
                        ground_truth_index,
                        prediction_index,
                    )
                )

    candidates.sort(reverse=True)
    used_ground_truth = set()
    used_predictions = set()
    matches = {}

    for overlap, confidence, ground_truth_index, prediction_index in candidates:
        if ground_truth_index in used_ground_truth:
            continue

        if prediction_index in used_predictions:
            continue

        used_ground_truth.add(ground_truth_index)
        used_predictions.add(prediction_index)
        matches[ground_truth_index] = (
            filtered_predictions[prediction_index],
            overlap,
            confidence,
        )

    observations = []

    for ground_truth_index, ground_truth in enumerate(frame["objects"]):
        match = matches.get(ground_truth_index)

        if match is None:
            predicted_class = None
            prediction_confidence = None
            match_iou = None
            status = "miss"
        else:
            prediction, match_iou, prediction_confidence = match
            predicted_class = prediction["class_name"]
            status = (
                "correct"
                if predicted_class == ground_truth["state"]
                else "misclassified"
            )

        observations.append(
            {
                "image_path": frame["image_path"],
                "sequence": frame["sequence"],
                "timestamp": frame["timestamp"],
                "track_id": ground_truth["track_id"],
                "track_key": ground_truth["track_key"],
                "ground_truth": ground_truth["state"],
                "status": status,
                "predicted_class": predicted_class,
                "confidence": prediction_confidence,
                "iou": match_iou,
                "correct": status == "correct",
            }
        )

    return observations


def build_tracks(frames, predictions_by_image, confidence_threshold):
    tracks = defaultdict(list)

    for image_path, frame in frames.items():
        predictions = predictions_by_image.get(image_path)

        if predictions is None:
            raise RuntimeError(f"Predictions missing for image: {image_path}")

        observations = match_frame(
            frame,
            predictions,
            confidence_threshold,
        )

        for observation in observations:
            tracks[observation["track_key"]].append(observation)

    for observations in tracks.values():
        observations.sort(key=lambda item: item["timestamp"])

    return tracks


def valid_window(observations, start_index, window_seconds):
    start_time = observations[start_index]["timestamp"]
    required_end_time = start_time + window_seconds

    if observations[-1]["timestamp"] < required_end_time:
        return None

    window = []

    for observation in observations[start_index:]:
        if observation["timestamp"] > required_end_time:
            break
        window.append(observation)

    if len(window) < MIN_STABILITY_OBSERVATIONS:
        return None

    for previous, current in zip(window, window[1:]):
        if (
            current["timestamp"] - previous["timestamp"]
            > MAX_CONSECUTIVE_GAP_SECONDS
        ):
            return None

    return window


def stability_summary(tracks, window_seconds):
    eligible_tracks = 0
    stable_tracks = 0
    first_correct_times = []
    stable_start_times = []
    stabilization_delays = []

    for observations in tracks.values():
        if len(observations) < MIN_STABILITY_OBSERVATIONS:
            continue

        track_start = observations[0]["timestamp"]
        first_correct = next(
            (
                observation["timestamp"]
                for observation in observations
                if observation["correct"]
            ),
            None,
        )

        has_eligible_window = False
        stable_start = None

        for start_index in range(len(observations)):
            window = valid_window(
                observations,
                start_index,
                window_seconds,
            )

            if window is None:
                continue

            has_eligible_window = True
            correct_count = sum(item["correct"] for item in window)
            success_rate = correct_count / len(window)

            if success_rate >= STABILITY_SUCCESS_RATE:
                stable_start = observations[start_index]["timestamp"]
                break

        if not has_eligible_window:
            continue

        eligible_tracks += 1

        if first_correct is not None:
            first_correct_times.append(first_correct - track_start)

        if stable_start is not None:
            stable_tracks += 1
            stable_start_times.append(stable_start - track_start)

            if first_correct is not None:
                stabilization_delays.append(stable_start - first_correct)

    return {
        "eligible_tracks": eligible_tracks,
        "stable_tracks": stable_tracks,
        "stable_rate": (
            stable_tracks / eligible_tracks if eligible_tracks else None
        ),
        "median_first_correct_seconds": rounded(
            median_or_none(first_correct_times)
        ),
        "mean_first_correct_seconds": rounded(mean_or_none(first_correct_times)),
        "median_stable_start_seconds": rounded(
            median_or_none(stable_start_times)
        ),
        "mean_stable_start_seconds": rounded(mean_or_none(stable_start_times)),
        "median_stabilization_delay_seconds": rounded(
            median_or_none(stabilization_delays)
        ),
        "mean_stabilization_delay_seconds": rounded(
            mean_or_none(stabilization_delays)
        ),
    }


def episode_summary(tracks, predicate):
    episodes = []

    for observations in tracks.values():
        current_episode = []

        def finish_episode():
            nonlocal current_episode

            if current_episode:
                episodes.append(
                    {
                        "frames": len(current_episode),
                        "duration_seconds": (
                            current_episode[-1]["timestamp"]
                            - current_episode[0]["timestamp"]
                        ),
                    }
                )
                current_episode = []

        for observation in observations:
            is_error = predicate(observation)

            if not is_error:
                finish_episode()
                continue

            if current_episode:
                gap = (
                    observation["timestamp"]
                    - current_episode[-1]["timestamp"]
                )

                if gap > MAX_CONSECUTIVE_GAP_SECONDS:
                    finish_episode()

            current_episode.append(observation)

        finish_episode()

    frame_counts = [episode["frames"] for episode in episodes]
    durations = [episode["duration_seconds"] for episode in episodes]

    return {
        "episode_count": len(episodes),
        "multi_frame_episode_count": sum(value >= 2 for value in frame_counts),
        "total_error_frames": sum(frame_counts),
        "maximum_consecutive_frames": max(frame_counts, default=0),
        "maximum_duration_seconds": rounded(max(durations, default=0.0)),
        "median_duration_seconds": rounded(median_or_none(durations)),
        "mean_duration_seconds": rounded(mean_or_none(durations)),
    }


def hazard_summary(tracks):
    all_observations = [
        observation
        for observations in tracks.values()
        for observation in observations
    ]
    red_observations = [
        observation
        for observation in all_observations
        if observation["ground_truth"] == "red"
    ]
    red_yellow_observations = [
        observation
        for observation in all_observations
        if observation["ground_truth"] == "red_yellow"
    ]

    red_miss_count = sum(
        observation["status"] == "miss"
        for observation in red_observations
    )
    red_to_green_count = sum(
        observation["predicted_class"] == "green"
        for observation in red_observations
    )
    red_hazard_count = sum(
        observation["status"] == "miss"
        or observation["predicted_class"] == "green"
        for observation in red_observations
    )
    red_yellow_to_green_count = sum(
        observation["predicted_class"] == "green"
        for observation in red_yellow_observations
    )

    red_count = len(red_observations)
    red_yellow_count = len(red_yellow_observations)

    return {
        "red_frame_count": red_count,
        "red_miss_count": red_miss_count,
        "red_miss_rate": red_miss_count / red_count if red_count else None,
        "red_to_green_count": red_to_green_count,
        "red_to_green_rate": (
            red_to_green_count / red_count if red_count else None
        ),
        "red_hazard_count": red_hazard_count,
        "red_hazard_rate": red_hazard_count / red_count if red_count else None,
        "red_yellow_frame_count": red_yellow_count,
        "red_yellow_to_green_count": red_yellow_to_green_count,
        "red_yellow_to_green_rate": (
            red_yellow_to_green_count / red_yellow_count
            if red_yellow_count
            else None
        ),
        "red_miss_episodes": episode_summary(
            tracks,
            lambda item: (
                item["ground_truth"] == "red" and item["status"] == "miss"
            ),
        ),
        "red_to_green_episodes": episode_summary(
            tracks,
            lambda item: (
                item["ground_truth"] == "red"
                and item["predicted_class"] == "green"
            ),
        ),
        "red_hazard_episodes": episode_summary(
            tracks,
            lambda item: (
                item["ground_truth"] == "red"
                and (
                    item["status"] == "miss"
                    or item["predicted_class"] == "green"
                )
            ),
        ),
    }


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 100)
    print("TEMPORAL STABILITY AND HAZARDOUS ERROR ANALYSIS")
    print("=" * 100)

    for path in PREDICTION_PATHS.values():
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")

    reference_predictions = load_predictions(PREDICTION_PATHS["yolov8n"])
    test_image_paths = set(reference_predictions)

    if len(test_image_paths) != 4244:
        raise RuntimeError(
            f"Expected 4244 test images, found {len(test_image_paths)}"
        )

    frames, missing_track_ids, invalid_boxes = load_ground_truth(
        test_image_paths
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = {}
    stability_rows = []
    hazard_rows = []

    for model_name, prediction_path in PREDICTION_PATHS.items():
        print()
        print(f"ANALYZING MODEL: {model_name}")
        predictions_by_image = load_predictions(prediction_path)

        if set(predictions_by_image) != test_image_paths:
            raise RuntimeError(
                f"Prediction image set differs for model: {model_name}"
            )

        model_result = {}

        for confidence_threshold in CONFIDENCE_THRESHOLDS:
            tracks = build_tracks(
                frames,
                predictions_by_image,
                confidence_threshold,
            )
            hazards = hazard_summary(tracks)
            stability_by_window = {}

            for window_seconds in STABILITY_WINDOWS:
                stability = stability_summary(tracks, window_seconds)
                stability_by_window[str(window_seconds)] = stability
                stability_rows.append(
                    {
                        "model": model_name,
                        "confidence": confidence_threshold,
                        "window_seconds": window_seconds,
                        **stability,
                    }
                )

            hazard_rows.append(
                {
                    "model": model_name,
                    "confidence": confidence_threshold,
                    "red_frame_count": hazards["red_frame_count"],
                    "red_miss_count": hazards["red_miss_count"],
                    "red_miss_rate": hazards["red_miss_rate"],
                    "red_to_green_count": hazards["red_to_green_count"],
                    "red_to_green_rate": hazards["red_to_green_rate"],
                    "red_hazard_count": hazards["red_hazard_count"],
                    "red_hazard_rate": hazards["red_hazard_rate"],
                    "max_hazard_frames": hazards[
                        "red_hazard_episodes"
                    ]["maximum_consecutive_frames"],
                    "max_hazard_seconds": hazards[
                        "red_hazard_episodes"
                    ]["maximum_duration_seconds"],
                }
            )

            model_result[str(confidence_threshold)] = {
                "track_count": len(tracks),
                "stability": stability_by_window,
                "hazards": hazards,
            }

        all_results[model_name] = model_result

    output_json = OUTPUT_ROOT / "temporal_analysis.json"
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "definitions": {
                    "primary_confidence": PRIMARY_CONFIDENCE,
                    "primary_stability_window_seconds": (
                        PRIMARY_STABILITY_WINDOW
                    ),
                    "match_iou": MATCH_IOU,
                    "minimum_stability_observations": (
                        MIN_STABILITY_OBSERVATIONS
                    ),
                    "stability_success_rate": STABILITY_SUCCESS_RATE,
                    "maximum_consecutive_gap_seconds": (
                        MAX_CONSECUTIVE_GAP_SECONDS
                    ),
                    "error_duration_definition": (
                        "Elapsed timestamp span from first to last erroneous "
                        "observation; a single-frame episode has span 0 seconds."
                    ),
                },
                "ground_truth_check": {
                    "test_images": len(frames),
                    "missing_track_ids": missing_track_ids,
                    "invalid_boxes_excluded": invalid_boxes,
                },
                "models": all_results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    write_csv(OUTPUT_ROOT / "stability_sensitivity.csv", stability_rows)
    write_csv(OUTPUT_ROOT / "hazard_sensitivity.csv", hazard_rows)

    print()
    print("=" * 100)
    print("PRIMARY RESULTS: confidence=0.25, stability window=2.0 seconds")
    print("=" * 100)

    for model_name in PREDICTION_PATHS:
        primary = all_results[model_name][str(PRIMARY_CONFIDENCE)]
        stability = primary["stability"][str(PRIMARY_STABILITY_WINDOW)]
        hazards = primary["hazards"]

        print()
        print(f"[{model_name}]")
        print(f"Tracks: {primary['track_count']}")
        print(
            f"Stable tracks: {stability['stable_tracks']}/"
            f"{stability['eligible_tracks']} "
            f"({(stability['stable_rate'] or 0) * 100:.2f}%)"
        )
        print(
            "Median stable start: "
            f"{stability['median_stable_start_seconds']} seconds"
        )
        print(
            f"Red misses: {hazards['red_miss_count']}/"
            f"{hazards['red_frame_count']} "
            f"({(hazards['red_miss_rate'] or 0) * 100:.2f}%)"
        )
        print(
            f"Red->Green: {hazards['red_to_green_count']}/"
            f"{hazards['red_frame_count']} "
            f"({(hazards['red_to_green_rate'] or 0) * 100:.4f}%)"
        )
        print(
            "Maximum hazardous episode: "
            f"{hazards['red_hazard_episodes']['maximum_consecutive_frames']} "
            "frames, "
            f"{hazards['red_hazard_episodes']['maximum_duration_seconds']} "
            "seconds"
        )

    print()
    print(f"JSON: {output_json}")
    print(f"CSV: {OUTPUT_ROOT / 'stability_sensitivity.csv'}")
    print(f"CSV: {OUTPUT_ROOT / 'hazard_sensitivity.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
