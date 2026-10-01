import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.temporal_robustness.repository_paths import DATASET_ROOT, DEGRADED_ROOT

CLEAN_DATASET_ROOT = DATASET_ROOT
CLEAN_IMAGE_ROOT = CLEAN_DATASET_ROOT / "images" / "test"
CLEAN_LABEL_ROOT = CLEAN_DATASET_ROOT / "labels" / "test"

OUTPUT_JSON = DEGRADED_ROOT / "degradation_validation.json"

CONDITIONS = [
    {"name": "fog_l1", "type": "fog", "value": 0.15},
    {"name": "fog_l2", "type": "fog", "value": 0.30},
    {"name": "fog_l3", "type": "fog", "value": 0.45},
    {"name": "lowlight_l1", "type": "lowlight", "value": 1.4},
    {"name": "lowlight_l2", "type": "lowlight", "value": 2.0},
    {"name": "lowlight_l3", "type": "lowlight", "value": 2.8},
    {"name": "motionblur_l1", "type": "motionblur", "value": 3},
    {"name": "motionblur_l2", "type": "motionblur", "value": 7},
    {"name": "motionblur_l3", "type": "motionblur", "value": 11},
]

EXPECTED_IMAGES = 4244
EXPECTED_WIDTH = 2048
EXPECTED_HEIGHT = 1024
EXACT_TRANSFORM_SAMPLE_COUNT = 30
RANDOM_SEED = 42


def apply_fog(array, alpha):
    result = array.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lowlight(array, gamma):
    normalized = array.astype(np.float32) / 255.0
    result = np.power(normalized, gamma) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_motionblur(array, kernel_size):
    kernel_size = int(kernel_size)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(array, -1, kernel)


def expected_transform(array, condition):
    if condition["type"] == "fog":
        return apply_fog(array, float(condition["value"]))
    if condition["type"] == "lowlight":
        return apply_lowlight(array, float(condition["value"]))
    if condition["type"] == "motionblur":
        return apply_motionblur(array, int(condition["value"]))

    raise ValueError(f"Unknown condition: {condition}")


def load_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def image_metrics(array):
    laplacian = cv2.Laplacian(array, cv2.CV_64F)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "laplacian_variance": float(np.var(laplacian)),
    }


def mean_metric(metric_rows, key):
    return sum(row[key] for row in metric_rows) / len(metric_rows)


def main():
    start_time = time.time()
    clean_images = sorted(CLEAN_IMAGE_ROOT.rglob("*.png"))
    clean_relative_paths = {
        path.relative_to(CLEAN_IMAGE_ROOT)
        for path in clean_images
    }
    clean_label_relative_paths = {
        path.relative_to(CLEAN_LABEL_ROOT)
        for path in CLEAN_LABEL_ROOT.rglob("*.txt")
    }

    if len(clean_images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} clean images, found {len(clean_images)}"
        )

    if clean_relative_paths != {
        path.with_suffix(".png")
        for path in clean_label_relative_paths
    }:
        raise RuntimeError("Clean Test image and label path sets do not match")

    rng = random.Random(RANDOM_SEED)
    sample_paths = rng.sample(
        sorted(clean_relative_paths),
        EXACT_TRANSFORM_SAMPLE_COUNT,
    )

    print("=" * 112)
    print("VALIDATE SYNTHETICALLY DEGRADED DTLD TEST SETS")
    print("=" * 112)
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Expected images per condition: {EXPECTED_IMAGES:,}")
    print(f"Full PNG integrity checks: {EXPECTED_IMAGES * len(CONDITIONS):,}")
    print(
        f"Exact pixel-transform samples per condition: "
        f"{EXACT_TRANSFORM_SAMPLE_COUNT}"
    )
    print("=" * 112)

    results = {}
    all_errors = []

    for condition_index, condition in enumerate(CONDITIONS, start=1):
        name = condition["name"]
        condition_root = DEGRADED_ROOT / name
        image_root = condition_root / "images" / "test"
        label_root = condition_root / "labels" / "test"
        data_yaml = condition_root / "data.yaml"

        print()
        print(
            f"[{condition_index}/{len(CONDITIONS)}] Validating {name} "
            f"({condition['type']}={condition['value']})"
        )

        image_paths = sorted(image_root.rglob("*.png"))
        label_paths = sorted(label_root.rglob("*.txt"))
        image_relative_paths = {
            path.relative_to(image_root)
            for path in image_paths
        }
        label_relative_paths = {
            path.relative_to(label_root)
            for path in label_paths
        }
        temporary_files = list(condition_root.rglob("*.temporary*"))
        condition_errors = []
        invalid_sizes = 0
        invalid_modes = 0
        corrupted_pngs = 0
        label_mismatches = 0

        if len(image_paths) != EXPECTED_IMAGES:
            condition_errors.append(
                f"Image count is {len(image_paths)}, expected {EXPECTED_IMAGES}"
            )
        if len(label_paths) != EXPECTED_IMAGES:
            condition_errors.append(
                f"Label count is {len(label_paths)}, expected {EXPECTED_IMAGES}"
            )
        if image_relative_paths != clean_relative_paths:
            condition_errors.append("Image relative path set differs from Clean")
        if label_relative_paths != clean_label_relative_paths:
            condition_errors.append("Label relative path set differs from Clean")
        if not data_yaml.exists():
            condition_errors.append("data.yaml is missing")
        if temporary_files:
            condition_errors.append(
                f"Temporary files remain: {len(temporary_files)}"
            )

        for image_index, image_path in enumerate(image_paths, start=1):
            try:
                with Image.open(image_path) as image:
                    if image.size != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
                        invalid_sizes += 1
                    if image.mode != "L":
                        invalid_modes += 1
                    image.verify()
            except Exception as error:
                corrupted_pngs += 1
                if corrupted_pngs <= 5:
                    condition_errors.append(
                        f"Corrupted PNG: {image_path} | {repr(error)}"
                    )

            relative_path = image_path.relative_to(image_root)
            degraded_label = (
                label_root / relative_path
            ).with_suffix(".txt")
            clean_label = (
                CLEAN_LABEL_ROOT / relative_path
            ).with_suffix(".txt")

            if (
                not degraded_label.exists()
                or degraded_label.read_bytes() != clean_label.read_bytes()
            ):
                label_mismatches += 1

            if image_index % 1000 == 0:
                print(
                    f"  PNG checks: {image_index:,} / {len(image_paths):,}"
                )

        exact_mismatches = 0
        maximum_absolute_difference = 0
        clean_metric_rows = []
        degraded_metric_rows = []

        for relative_path in sample_paths:
            clean_path = CLEAN_IMAGE_ROOT / relative_path
            degraded_path = image_root / relative_path
            clean_array = load_array(clean_path)
            degraded_array = load_array(degraded_path)
            expected_array = expected_transform(clean_array, condition)

            difference = np.abs(
                degraded_array.astype(np.int16)
                - expected_array.astype(np.int16)
            )
            sample_maximum = int(np.max(difference))
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                sample_maximum,
            )

            if not np.array_equal(degraded_array, expected_array):
                exact_mismatches += 1

            clean_metric_rows.append(image_metrics(clean_array))
            degraded_metric_rows.append(image_metrics(degraded_array))

        if invalid_sizes:
            condition_errors.append(f"Invalid image sizes: {invalid_sizes}")
        if invalid_modes:
            condition_errors.append(f"Invalid image modes: {invalid_modes}")
        if corrupted_pngs:
            condition_errors.append(f"Corrupted PNG count: {corrupted_pngs}")
        if label_mismatches:
            condition_errors.append(f"Label mismatches: {label_mismatches}")
        if exact_mismatches:
            condition_errors.append(
                f"Exact transform mismatches: {exact_mismatches}"
            )

        metrics = {
            "clean_mean": mean_metric(clean_metric_rows, "mean"),
            "degraded_mean": mean_metric(degraded_metric_rows, "mean"),
            "clean_standard_deviation": mean_metric(
                clean_metric_rows,
                "standard_deviation",
            ),
            "degraded_standard_deviation": mean_metric(
                degraded_metric_rows,
                "standard_deviation",
            ),
            "clean_laplacian_variance": mean_metric(
                clean_metric_rows,
                "laplacian_variance",
            ),
            "degraded_laplacian_variance": mean_metric(
                degraded_metric_rows,
                "laplacian_variance",
            ),
        }

        result = {
            "condition": condition,
            "image_count": len(image_paths),
            "label_count": len(label_paths),
            "relative_image_paths_match_clean": (
                image_relative_paths == clean_relative_paths
            ),
            "relative_label_paths_match_clean": (
                label_relative_paths == clean_label_relative_paths
            ),
            "invalid_image_sizes": invalid_sizes,
            "invalid_image_modes": invalid_modes,
            "corrupted_pngs": corrupted_pngs,
            "label_mismatches": label_mismatches,
            "temporary_files": len(temporary_files),
            "exact_transform_samples": EXACT_TRANSFORM_SAMPLE_COUNT,
            "exact_transform_mismatches": exact_mismatches,
            "maximum_absolute_pixel_difference": maximum_absolute_difference,
            "sample_metrics": metrics,
            "errors": condition_errors,
            "passed": not condition_errors,
        }
        results[name] = result
        all_errors.extend(
            {"condition": name, "error": error}
            for error in condition_errors
        )

        print(
            f"  Result: {'PASS' if result['passed'] else 'FAIL'} | "
            f"images={len(image_paths):,}, labels={len(label_paths):,}, "
            f"bad PNG={corrupted_pngs}, label mismatch={label_mismatches}, "
            f"transform mismatch={exact_mismatches}"
        )
        print(
            f"  Sample metrics: mean {metrics['clean_mean']:.2f} -> "
            f"{metrics['degraded_mean']:.2f}, std "
            f"{metrics['clean_standard_deviation']:.2f} -> "
            f"{metrics['degraded_standard_deviation']:.2f}, "
            f"Laplacian {metrics['clean_laplacian_variance']:.2f} -> "
            f"{metrics['degraded_laplacian_variance']:.2f}"
        )

    elapsed_seconds = time.time() - start_time
    passed_conditions = sum(result["passed"] for result in results.values())

    output = {
        "definitions": {
            "clean_dataset_root": str(CLEAN_DATASET_ROOT),
            "degraded_root": str(DEGRADED_ROOT),
            "expected_images_per_condition": EXPECTED_IMAGES,
            "expected_width": EXPECTED_WIDTH,
            "expected_height": EXPECTED_HEIGHT,
            "exact_transform_sample_count": EXACT_TRANSFORM_SAMPLE_COUNT,
            "random_seed": RANDOM_SEED,
        },
        "passed_conditions": passed_conditions,
        "condition_count": len(CONDITIONS),
        "total_errors": len(all_errors),
        "errors": all_errors,
        "results": results,
        "elapsed_seconds": elapsed_seconds,
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 112)
    print("DEGRADED TEST SET VALIDATION RESULT")
    print("=" * 112)
    print(f"Passed conditions: {passed_conditions} / {len(CONDITIONS)}")
    print(f"Total errors: {len(all_errors)}")
    print(f"Elapsed time: {elapsed_seconds / 60.0:.1f} minutes")
    print(f"JSON: {OUTPUT_JSON}")
    print("=" * 112)

    if all_errors:
        raise RuntimeError(
            f"Validation failed with {len(all_errors)} errors. "
            f"Check {OUTPUT_JSON}"
        )


if __name__ == "__main__":
    main()
