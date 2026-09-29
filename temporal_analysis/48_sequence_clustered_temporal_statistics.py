import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.project_paths import OUTPUTS_ROOT, RESULTS_ROOT

INPUT_ROOT = Path(
    os.environ.get(
        "DTLD_CORRECTED_TEMPORAL_INPUT_ROOT",
        str(OUTPUTS_ROOT / "corrected_temporal_analysis"),
    )
)
if not (INPUT_ROOT / "stability_track_metrics_corrected.csv").exists():
    INPUT_ROOT = RESULTS_ROOT / "temporal" / "track_metrics"
OUTPUT_ROOT = Path(
    os.environ.get(
        "DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT",
        str(OUTPUTS_ROOT / "sequence_clustered_statistics"),
    )
)

STABILITY_CSV = INPUT_ROOT / "stability_track_metrics_corrected.csv"
RED_CSV = INPUT_ROOT / "red_track_metrics_corrected.csv"

MODELS = ("yolov8n", "yolo11n", "yolo12n")
MODEL_PAIRS = (
    ("yolov8n", "yolo11n"),
    ("yolov8n", "yolo12n"),
    ("yolo11n", "yolo12n"),
)
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
REPORT_CONDITIONS = ("clean", "fog_l3", "lowlight_l3", "motionblur_l3")
SEVERE_CONDITIONS = ("fog_l3", "lowlight_l3", "motionblur_l3")

AUC_HORIZON_SECONDS = 5.0
PERMUTATION_ITERATIONS = int(
    os.environ.get("DTLD_CLUSTER_PERMUTATION_ITERATIONS", "100000")
)
RANDOM_SEED = 42
ANALYSIS_CODES = {"direct": 0, "clean_change": 1}
METRIC_CODES = {"auc5": 0, "red_duration": 1}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_stability():
    data = defaultdict(lambda: defaultdict(dict))
    for row in read_csv(STABILITY_CSV):
        start_text = row["stable_start_seconds"].strip()
        data[row["condition"]][row["model"]][row["track_key"]] = {
            "eligible": int(row["eligible"]),
            "stable": int(row["stable"]),
            "start": float(start_text) if start_text else None,
        }
    return data


def load_red():
    data = defaultdict(lambda: defaultdict(dict))
    for row in read_csv(RED_CSV):
        data[row["condition"]][row["model"]][row["track_key"]] = float(
            row["longest_complete_miss_seconds"]
        )
    return data


def common_keys(*mappings):
    return sorted(set.intersection(*(set(mapping) for mapping in mappings)))


def common_eligible_keys(stability, condition, models=MODELS):
    mappings = []
    for model in models:
        mappings.append(
            {
                key: values
                for key, values in stability[condition][model].items()
                if values["eligible"] == 1
            }
        )
    return common_keys(*mappings)


def sequence_key(track_key):
    parts = track_key.replace("\\", "/").split("/")
    if len(parts) != 4:
        raise ValueError(f"Unexpected track_key hierarchy: {track_key}")
    return "/".join(parts[:3])


def auc_contribution(value):
    if value["stable"] != 1 or value["start"] is None:
        return 0.0
    start = value["start"]
    if start > AUC_HORIZON_SECONDS:
        return 0.0
    return max(0.0, (AUC_HORIZON_SECONDS - start) / AUC_HORIZON_SECONDS)


def cluster_sums_counts(values, track_keys):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(track_keys):
        raise ValueError("values and track_keys must have the same length")
    if len(values) == 0:
        raise ValueError("Cannot test an empty sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("All tested values must be finite")

    grouped = defaultdict(list)
    for value, key in zip(values, track_keys):
        grouped[sequence_key(key)].append(float(value))
    cluster_names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in cluster_names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in cluster_names], dtype=np.int64)
    return cluster_names, sums, counts


def cluster_sign_flip_p(values, track_keys, rng):
    values = np.asarray(values, dtype=np.float64)
    clusters, sums, counts = cluster_sums_counts(values, track_keys)
    observed = abs(float(values.mean()))
    if observed <= 1e-15:
        return 1.0, len(clusters)

    total_count = int(counts.sum())
    extreme = 0
    chunk = 500
    for start in range(0, PERMUTATION_ITERATIONS, chunk):
        count = min(chunk, PERMUTATION_ITERATIONS - start)
        signs = rng.integers(0, 2, size=(count, len(clusters))) * 2 - 1
        permuted = (signs * sums).sum(axis=1) / total_count
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed))
    p_value = float((extreme + 1) / (PERMUTATION_ITERATIONS + 1))
    return p_value, len(clusters)


def permutation_rng(analysis, metric, condition_index, pair_index):
    """Return an order-independent RNG stream for one reported test."""
    seed = np.random.SeedSequence(
        [
            RANDOM_SEED,
            ANALYSIS_CODES[analysis],
            METRIC_CODES[metric],
            condition_index,
            pair_index,
        ]
    )
    return np.random.Generator(np.random.PCG64(seed))


def holm_adjust(rows, raw_key, output_key):
    indexed = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get(raw_key) is not None
    ]
    ordered = sorted(indexed, key=lambda item: item[1][raw_key])
    total = len(ordered)
    running = 0.0
    adjusted = {}
    for rank, (original_index, row) in enumerate(ordered):
        candidate = min(1.0, row[raw_key] * (total - rank))
        running = max(running, candidate)
        adjusted[original_index] = running
    for index, row in enumerate(rows):
        row[output_key] = adjusted.get(index)


def build_summary(stability, red):
    rows = []
    for condition in CONDITIONS:
        stable_keys = common_eligible_keys(stability, condition)
        red_keys = common_keys(*(red[condition][model] for model in MODELS))
        for model in MODELS:
            auc_values = [
                auc_contribution(stability[condition][model][key])
                for key in stable_keys
            ]
            red_values = [red[condition][model][key] for key in red_keys]
            auc_clusters = len(cluster_sums_counts(auc_values, stable_keys)[0])
            red_clusters = len(cluster_sums_counts(red_values, red_keys)[0])
            rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "auc5_tracks": len(stable_keys),
                    "auc5_sequence_clusters": auc_clusters,
                    "auc5": float(np.mean(auc_values)) * 100.0,
                    "red_tracks": len(red_keys),
                    "red_sequence_clusters": red_clusters,
                    "mean_longest_red_miss_seconds": float(np.mean(red_values)),
                }
            )
    return rows


def build_direct(stability, red):
    rows = []
    for condition_index, condition in enumerate(REPORT_CONDITIONS):
        for pair_index, (model_a, model_b) in enumerate(MODEL_PAIRS):
            stable_keys = common_eligible_keys(stability, condition, (model_a, model_b))
            auc_a = np.asarray(
                [auc_contribution(stability[condition][model_a][key]) for key in stable_keys]
            )
            auc_b = np.asarray(
                [auc_contribution(stability[condition][model_b][key]) for key in stable_keys]
            )
            auc_diff = auc_b - auc_a
            auc_mean = float(auc_diff.mean())
            auc_p, auc_clusters = cluster_sign_flip_p(
                auc_diff,
                stable_keys,
                permutation_rng(
                    "direct", "auc5", condition_index, pair_index
                ),
            )

            red_keys = common_keys(red[condition][model_a], red[condition][model_b])
            red_a = np.asarray([red[condition][model_a][key] for key in red_keys])
            red_b = np.asarray([red[condition][model_b][key] for key in red_keys])
            red_diff = red_b - red_a
            red_mean = float(red_diff.mean())
            red_p, red_clusters = cluster_sign_flip_p(
                red_diff,
                red_keys,
                permutation_rng(
                    "direct", "red_duration", condition_index, pair_index
                ),
            )
            rows.append(
                {
                    "condition": condition,
                    "comparison": f"{model_b} - {model_a}",
                    "auc5_tracks": len(stable_keys),
                    "auc5_sequence_clusters": auc_clusters,
                    "auc5_difference_points": auc_mean * 100.0,
                    "auc5_cluster_permutation_p": auc_p,
                    "auc5_cluster_holm_p_global": None,
                    "red_tracks": len(red_keys),
                    "red_sequence_clusters": red_clusters,
                    "red_duration_difference_seconds": red_mean,
                    "red_duration_cluster_permutation_p": red_p,
                    "red_duration_cluster_holm_p_global": None,
                }
            )

    holm_adjust(rows, "auc5_cluster_permutation_p", "auc5_cluster_holm_p_global")
    holm_adjust(
        rows,
        "red_duration_cluster_permutation_p",
        "red_duration_cluster_holm_p_global",
    )
    return rows


def build_robustness(stability, red):
    rows = []
    for condition_index, condition in enumerate(SEVERE_CONDITIONS):
        for pair_index, (model_a, model_b) in enumerate(MODEL_PAIRS):
            stable_keys = common_keys(
                {
                    key: value
                    for key, value in stability["clean"][model_a].items()
                    if value["eligible"] == 1
                },
                {
                    key: value
                    for key, value in stability["clean"][model_b].items()
                    if value["eligible"] == 1
                },
                {
                    key: value
                    for key, value in stability[condition][model_a].items()
                    if value["eligible"] == 1
                },
                {
                    key: value
                    for key, value in stability[condition][model_b].items()
                    if value["eligible"] == 1
                },
            )
            auc_did = np.asarray(
                [
                    (
                        auc_contribution(stability[condition][model_b][key])
                        - auc_contribution(stability["clean"][model_b][key])
                    )
                    - (
                        auc_contribution(stability[condition][model_a][key])
                        - auc_contribution(stability["clean"][model_a][key])
                    )
                    for key in stable_keys
                ]
            )
            auc_mean = float(auc_did.mean())
            auc_p, auc_clusters = cluster_sign_flip_p(
                auc_did,
                stable_keys,
                permutation_rng(
                    "clean_change", "auc5", condition_index, pair_index
                ),
            )

            red_keys = common_keys(
                red["clean"][model_a],
                red["clean"][model_b],
                red[condition][model_a],
                red[condition][model_b],
            )
            red_did = np.asarray(
                [
                    (
                        red[condition][model_b][key] - red["clean"][model_b][key]
                    )
                    - (
                        red[condition][model_a][key] - red["clean"][model_a][key]
                    )
                    for key in red_keys
                ]
            )
            red_mean = float(red_did.mean())
            red_p, red_clusters = cluster_sign_flip_p(
                red_did,
                red_keys,
                permutation_rng(
                    "clean_change",
                    "red_duration",
                    condition_index,
                    pair_index,
                ),
            )
            rows.append(
                {
                    "condition": condition,
                    "comparison": f"{model_b} - {model_a}",
                    "auc5_tracks": len(stable_keys),
                    "auc5_sequence_clusters": auc_clusters,
                    "auc5_change_difference_points": auc_mean * 100.0,
                    "auc5_cluster_permutation_p": auc_p,
                    "auc5_cluster_holm_p_global": None,
                    "red_tracks": len(red_keys),
                    "red_sequence_clusters": red_clusters,
                    "red_duration_change_difference_seconds": red_mean,
                    "red_duration_cluster_permutation_p": red_p,
                    "red_duration_cluster_holm_p_global": None,
                }
            )

    holm_adjust(rows, "auc5_cluster_permutation_p", "auc5_cluster_holm_p_global")
    holm_adjust(
        rows,
        "red_duration_cluster_permutation_p",
        "red_duration_cluster_holm_p_global",
    )
    return rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stability = load_stability()
    red = load_red()

    summary = build_summary(stability, red)
    direct = build_direct(stability, red)
    robustness = build_robustness(stability, red)

    summary_path = OUTPUT_ROOT / "sequence_clustered_condition_summary.csv"
    direct_path = OUTPUT_ROOT / "sequence_clustered_direct_model_comparisons.csv"
    robustness_path = OUTPUT_ROOT / "sequence_clustered_robustness_comparisons.csv"
    json_path = OUTPUT_ROOT / "sequence_clustered_statistics.json"

    write_csv(summary_path, summary)
    write_csv(direct_path, direct)
    write_csv(robustness_path, robustness)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "settings": {
                    "cluster_unit": "city/route/intersection-passage sequence",
                    "permutation_iterations": PERMUTATION_ITERATIONS,
                    "permutation_base_seed": RANDOM_SEED,
                    "bit_generator": "PCG64",
                    "seed_scheme": (
                        "SeedSequence([base_seed, analysis_code, metric_code, "
                        "condition_index, model_pair_index])"
                    ),
                    "permutation": (
                        "two-sided cluster sign-flip: one sign per sequence "
                        "cluster; track-weighted statistic"
                    ),
                    "holm_scope": "direct and Clean-change families, separately by metric",
                },
                "summary": summary,
                "direct_comparisons": direct,
                "robustness_comparisons": robustness,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Sequence-clustered analysis completed: {OUTPUT_ROOT}")
    print(
        "Two-sided sequence-cluster sign-flip permutations per test: "
        f"{PERMUTATION_ITERATIONS:,}"
    )


if __name__ == "__main__":
    main()
