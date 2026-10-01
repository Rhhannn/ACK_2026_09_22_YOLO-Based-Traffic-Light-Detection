import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.temporal_robustness.repository_paths import DATASET_ROOT, DEGRADED_ROOT

CLEAN_DATASET_ROOT = DATASET_ROOT
CLEAN_IMAGE_ROOT = CLEAN_DATASET_ROOT / "images" / "test"
CLEAN_LABEL_ROOT = CLEAN_DATASET_ROOT / "labels" / "test"

OUTPUT_ROOT = DEGRADED_ROOT

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

EXPECTED_TEST_IMAGES = 4244
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1024
PNG_COMPRESSION_LEVEL = 3
PROGRESS_INTERVAL = 100


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


def apply_condition(array, condition):
    condition_type = condition["type"]
    value = condition["value"]

    if condition_type == "fog":
        return apply_fog(array, float(value))
    if condition_type == "lowlight":
        return apply_lowlight(array, float(value))
    if condition_type == "motionblur":
        return apply_motionblur(array, int(value))

    raise ValueError(f"Unknown condition type: {condition_type}")


def yaml_text(condition_root):
    root_text = condition_root.as_posix()
    clean_root_text = CLEAN_DATASET_ROOT.as_posix()

    return (
        f"path: {root_text}\n"
        f"train: {clean_root_text}/images/train\n"
        f"val: {clean_root_text}/images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: red\n"
        "  1: yellow\n"
        "  2: green\n"
        "  3: red_yellow\n"
    )


def atomic_save_png(array, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".temporary.png")
    Image.fromarray(array, mode="L").save(
        temporary,
        format="PNG",
        compress_level=PNG_COMPRESSION_LEVEL,
    )
    temporary.replace(destination)


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".temporary")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def prepare_condition_directories():
    for condition in CONDITIONS:
        condition_root = OUTPUT_ROOT / condition["name"]
        (condition_root / "images" / "test").mkdir(
            parents=True,
            exist_ok=True,
        )
        (condition_root / "labels" / "test").mkdir(
            parents=True,
            exist_ok=True,
        )
        data_yaml = condition_root / "data.yaml"
        data_yaml.write_text(
            yaml_text(condition_root),
            encoding="utf-8",
        )


def check_inputs(image_paths):
    if len(image_paths) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} Test images, "
            f"found {len(image_paths)}"
        )

    missing_labels = []

    for image_path in image_paths:
        relative_path = image_path.relative_to(CLEAN_IMAGE_ROOT)
        label_path = (CLEAN_LABEL_ROOT / relative_path).with_suffix(".txt")

        if not label_path.exists():
            missing_labels.append(str(label_path))

    if missing_labels:
        raise RuntimeError(
            f"Missing {len(missing_labels)} labels. "
            f"Examples: {missing_labels[:5]}"
        )


def check_disk_space(image_paths):
    clean_total_bytes = sum(path.stat().st_size for path in image_paths)
    estimated_bytes = clean_total_bytes * len(CONDITIONS)
    required_bytes = int(estimated_bytes * 1.5) + 10 * 1024**3
    free_bytes = shutil.disk_usage(OUTPUT_ROOT.parent).free

    print(f"Clean Test size: {clean_total_bytes / 1024**3:.2f} GB")
    print(
        f"Estimated nine-set size: "
        f"{estimated_bytes / 1024**3:.2f} GB"
    )
    print(f"Available disk space: {free_bytes / 1024**3:.2f} GB")

    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough free disk space for safe dataset generation"
        )


def main():
    start_time = time.time()
    image_paths = sorted(CLEAN_IMAGE_ROOT.rglob("*.png"))

    print("=" * 110)
    print("BUILD SYNTHETICALLY DEGRADED DTLD TEST SETS")
    print("=" * 110)
    print(f"Clean Test root: {CLEAN_IMAGE_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Clean Test images: {len(image_paths)}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Expected generated images: {len(image_paths) * len(CONDITIONS):,}")
    print("=" * 110)

    check_inputs(image_paths)
    check_disk_space(image_paths)
    prepare_condition_directories()

    statistics = {
        condition["name"]: {
            "generated_images": 0,
            "skipped_existing_images": 0,
            "copied_labels": 0,
            "skipped_existing_labels": 0,
        }
        for condition in CONDITIONS
    }
    errors = []

    for image_index, clean_image_path in enumerate(image_paths, start=1):
        relative_image_path = clean_image_path.relative_to(CLEAN_IMAGE_ROOT)
        clean_label_path = (
            CLEAN_LABEL_ROOT / relative_image_path
        ).with_suffix(".txt")

        try:
            with Image.open(clean_image_path) as image:
                clean_array = np.asarray(image.convert("L"), dtype=np.uint8)

            if clean_array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise RuntimeError(
                    f"Unexpected image shape {clean_array.shape}: "
                    f"{clean_image_path}"
                )

            for condition in CONDITIONS:
                condition_name = condition["name"]
                condition_root = OUTPUT_ROOT / condition_name
                output_image_path = (
                    condition_root / "images" / "test" / relative_image_path
                )
                output_label_path = (
                    condition_root
                    / "labels"
                    / "test"
                    / relative_image_path
                ).with_suffix(".txt")

                if output_image_path.exists():
                    statistics[condition_name][
                        "skipped_existing_images"
                    ] += 1
                else:
                    degraded_array = apply_condition(clean_array, condition)
                    atomic_save_png(degraded_array, output_image_path)
                    statistics[condition_name]["generated_images"] += 1

                if output_label_path.exists():
                    statistics[condition_name][
                        "skipped_existing_labels"
                    ] += 1
                else:
                    atomic_copy(clean_label_path, output_label_path)
                    statistics[condition_name]["copied_labels"] += 1

        except Exception as error:
            errors.append(
                {
                    "image_path": str(clean_image_path),
                    "error": repr(error),
                }
            )
            print(f"ERROR: {clean_image_path}")
            print(repr(error))

        if image_index % PROGRESS_INTERVAL == 0 or image_index == len(image_paths):
            elapsed_minutes = (time.time() - start_time) / 60.0
            print(
                f"Progress: {image_index:,} / {len(image_paths):,} "
                f"Clean images | elapsed {elapsed_minutes:.1f} minutes"
            )

    final_counts = {}

    for condition in CONDITIONS:
        condition_name = condition["name"]
        condition_root = OUTPUT_ROOT / condition_name
        image_count = len(
            list((condition_root / "images" / "test").rglob("*.png"))
        )
        label_count = len(
            list((condition_root / "labels" / "test").rglob("*.txt"))
        )
        final_counts[condition_name] = {
            "image_count": image_count,
            "label_count": label_count,
        }

    elapsed_seconds = time.time() - start_time
    summary = {
        "clean_dataset_root": str(CLEAN_DATASET_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "clean_test_image_count": len(image_paths),
        "expected_images_per_condition": EXPECTED_TEST_IMAGES,
        "png_compression_level": PNG_COMPRESSION_LEVEL,
        "conditions": CONDITIONS,
        "statistics": statistics,
        "final_counts": final_counts,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
    }

    summary_path = OUTPUT_ROOT / "degradation_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("DEGRADED TEST SET BUILD RESULT")
    print("=" * 110)

    for condition in CONDITIONS:
        name = condition["name"]
        counts = final_counts[name]
        stats = statistics[name]
        print(
            f"{name:<16} "
            f"images={counts['image_count']:,} "
            f"labels={counts['label_count']:,} "
            f"new={stats['generated_images']:,} "
            f"skipped={stats['skipped_existing_images']:,}"
        )

    print(f"Errors: {len(errors)}")
    print(f"Elapsed time: {elapsed_seconds / 60.0:.1f} minutes")
    print(f"Summary: {summary_path}")
    print("=" * 110)

    if errors:
        raise RuntimeError(
            f"Build completed with {len(errors)} errors. Check the summary file."
        )

    for condition_name, counts in final_counts.items():
        if counts["image_count"] != EXPECTED_TEST_IMAGES:
            raise RuntimeError(
                f"{condition_name}: expected {EXPECTED_TEST_IMAGES} images, "
                f"found {counts['image_count']}"
            )
        if counts["label_count"] != EXPECTED_TEST_IMAGES:
            raise RuntimeError(
                f"{condition_name}: expected {EXPECTED_TEST_IMAGES} labels, "
                f"found {counts['label_count']}"
            )


if __name__ == "__main__":
    main()
