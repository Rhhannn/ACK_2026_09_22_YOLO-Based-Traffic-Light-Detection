import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.project_paths import DATASET_ROOT, OUTPUTS_ROOT

IMAGE_ROOT = DATASET_ROOT / "images" / "test"
LABEL_ROOT = DATASET_ROOT / "labels" / "test"
OUTPUT_ROOT = OUTPUTS_ROOT / "degradation_previews"

IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1024

TARGET_RED_HEIGHTS = [8, 20, 45]

FOG_LEVELS = [0.15, 0.30, 0.45]
LOW_LIGHT_GAMMAS = [1.4, 2.0, 2.8]
MOTION_BLUR_KERNELS = [3, 7, 11]

FULL_CELL_WIDTH = 768
FULL_CELL_HEIGHT = 384
CROP_CELL_SIZE = 360
HEADER_HEIGHT = 34


def read_red_candidates():
    candidates = []

    for label_path in LABEL_ROOT.rglob("*.txt"):
        relative_path = label_path.relative_to(LABEL_ROOT)
        image_path = (IMAGE_ROOT / relative_path).with_suffix(".png")

        if not image_path.exists():
            continue

        with label_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                if class_id != 0:
                    continue

                x_center, y_center, width, height = map(float, parts[1:])
                height_pixels = height * IMAGE_HEIGHT
                width_pixels = width * IMAGE_WIDTH
                x1 = (x_center - width / 2.0) * IMAGE_WIDTH
                y1 = (y_center - height / 2.0) * IMAGE_HEIGHT
                x2 = x1 + width_pixels
                y2 = y1 + height_pixels

                candidates.append(
                    {
                        "image_path": image_path,
                        "label_path": label_path,
                        "height_pixels": height_pixels,
                        "box": [x1, y1, x2, y2],
                    }
                )

    if not candidates:
        raise RuntimeError("No Red samples were found in the Test dataset")

    return candidates


def select_samples(candidates):
    selected = []
    used_images = set()

    for target_height in TARGET_RED_HEIGHTS:
        ranked = sorted(
            candidates,
            key=lambda item: abs(item["height_pixels"] - target_height),
        )

        chosen = next(
            (
                item
                for item in ranked
                if item["image_path"] not in used_images
            ),
            None,
        )

        if chosen is None:
            raise RuntimeError("Could not select three distinct Red samples")

        selected.append(chosen)
        used_images.add(chosen["image_path"])

    return selected


def apply_fog(array, alpha):
    result = array.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_low_light(array, gamma):
    normalized = array.astype(np.float32) / 255.0
    result = np.power(normalized, gamma) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_motion_blur(array, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(array, -1, kernel)


def resize_box(box, source_width, source_height, target_width, target_height):
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return [
        int(round(box[0] * scale_x)),
        int(round(box[1] * scale_y)),
        int(round(box[2] * scale_x)),
        int(round(box[3] * scale_y)),
    ]


def crop_region(image, box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)

    crop_width = max(180.0, box_width * 14.0)
    crop_height = max(180.0, box_height * 10.0)
    crop_size = min(max(crop_width, crop_height), 700.0)

    left = int(round(center_x - crop_size / 2.0))
    top = int(round(center_y - crop_size / 2.0))
    right = left + int(round(crop_size))
    bottom = top + int(round(crop_size))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image.width:
        left -= right - image.width
        right = image.width
    if bottom > image.height:
        top -= bottom - image.height
        bottom = image.height

    left = max(0, left)
    top = max(0, top)
    crop = image.crop((left, top, right, bottom))
    adjusted_box = [x1 - left, y1 - top, x2 - left, y2 - top]
    return crop, adjusted_box


def prepare_display_image(image, box, crop_mode):
    if crop_mode:
        source, adjusted_box = crop_region(image, box)
        target_width = CROP_CELL_SIZE
        target_height = CROP_CELL_SIZE
    else:
        source = image
        adjusted_box = box
        target_width = FULL_CELL_WIDTH
        target_height = FULL_CELL_HEIGHT

    resized = source.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    ).convert("RGB")
    scaled_box = resize_box(
        adjusted_box,
        source.width,
        source.height,
        target_width,
        target_height,
    )
    return resized, scaled_box


def draw_cell(image, box, title, width, height):
    cell = Image.new("RGB", (width, height + HEADER_HEIGHT), color=(28, 28, 28))
    cell.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(cell)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill=(255, 255, 255), font=font)

    shifted_box = [
        box[0],
        box[1] + HEADER_HEIGHT,
        box[2],
        box[3] + HEADER_HEIGHT,
    ]
    draw.rectangle(shifted_box, outline=(255, 64, 64), width=3)
    return cell


def make_sheet(samples, condition_name, level_names, transform, crop_mode):
    cell_width = CROP_CELL_SIZE if crop_mode else FULL_CELL_WIDTH
    cell_height = CROP_CELL_SIZE if crop_mode else FULL_CELL_HEIGHT
    columns = 4
    rows = len(samples)
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * (cell_height + HEADER_HEIGHT),
        ),
        color=(28, 28, 28),
    )

    for row_index, sample in enumerate(samples):
        clean_image = Image.open(sample["image_path"]).convert("L")
        clean_array = np.asarray(clean_image, dtype=np.uint8)
        arrays = [clean_array] + [transform(clean_array, level) for level in level_names]
        column_titles = ["Clean"] + [
            f"L{index}: {level}"
            for index, level in enumerate(level_names, start=1)
        ]

        for column_index, (array, column_title) in enumerate(
            zip(arrays, column_titles)
        ):
            processed = Image.fromarray(array, mode="L")
            display, display_box = prepare_display_image(
                processed,
                sample["box"],
                crop_mode,
            )
            title = (
                f"{condition_name} | {column_title} | "
                f"GT Red height={sample['height_pixels']:.1f}px"
            )
            cell = draw_cell(
                display,
                display_box,
                title,
                cell_width,
                cell_height,
            )
            sheet.paste(
                cell,
                (
                    column_index * cell_width,
                    row_index * (cell_height + HEADER_HEIGHT),
                ),
            )

    suffix = "crops" if crop_mode else "full"
    output_path = OUTPUT_ROOT / f"{condition_name.lower()}_levels_{suffix}.png"
    sheet.save(output_path)
    return output_path


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = read_red_candidates()
    samples = select_samples(candidates)

    print("=" * 100)
    print("DEGRADATION PREVIEW GENERATION")
    print("=" * 100)
    print("Selected Red samples:")

    for index, sample in enumerate(samples, start=1):
        print(
            f"{index}. height={sample['height_pixels']:.1f}px | "
            f"{sample['image_path']}"
        )

    outputs = []

    for crop_mode in [False, True]:
        outputs.append(
            make_sheet(
                samples,
                "Fog",
                FOG_LEVELS,
                apply_fog,
                crop_mode,
            )
        )
        outputs.append(
            make_sheet(
                samples,
                "LowLight",
                LOW_LIGHT_GAMMAS,
                apply_low_light,
                crop_mode,
            )
        )
        outputs.append(
            make_sheet(
                samples,
                "MotionBlur",
                MOTION_BLUR_KERNELS,
                apply_motion_blur,
                crop_mode,
            )
        )

    metadata_path = OUTPUT_ROOT / "degradation_preview_settings.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "fog_alpha_levels": FOG_LEVELS,
                "low_light_gamma_levels": LOW_LIGHT_GAMMAS,
                "motion_blur_kernel_levels": MOTION_BLUR_KERNELS,
                "selected_samples": [
                    {
                        "image_path": str(sample["image_path"]),
                        "label_path": str(sample["label_path"]),
                        "red_height_pixels": sample["height_pixels"],
                        "box_xyxy": sample["box"],
                    }
                    for sample in samples
                ],
                "notes": {
                    "fog": "Fixed white atmospheric overlay; no random frame variation",
                    "low_light": "Fixed gamma darkening; gamma greater than 1 is darker",
                    "motion_blur": "Fixed horizontal line kernel; no random frame variation",
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Generated preview files:")
    for output_path in outputs:
        print(output_path)
    print(metadata_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
