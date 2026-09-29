import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from common.project_paths import DATASET_ROOT, LABEL_JSON, RAW_DATA_ROOT, SPLIT_DIR

DATA_ROOT = RAW_DATA_ROOT
JSON_PATH = LABEL_JSON
SPLIT_MANIFEST_DIR = SPLIT_DIR
OUTPUT_ROOT = DATASET_ROOT

METADATA_DIR = OUTPUT_ROOT / "metadata"

CLASS_IDS = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "red_yellow": 3,
}

CLASS_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "red_yellow",
}

MAX_12BIT_VALUE = 4095.0


def load_split_sessions():
    split_sessions = {}

    for split in ("train", "val", "test"):
        manifest_path = (
            SPLIT_MANIFEST_DIR
            / f"{split}_sessions.txt"
        )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            sessions = {
                line.strip()
                for line in file
                if line.strip()
            }

        split_sessions[split] = sessions

    # 동일한 세션이 여러 split에 들어갔는지 검사한다.
    train_sessions = split_sessions["train"]
    val_sessions = split_sessions["val"]
    test_sessions = split_sessions["test"]

    if train_sessions & val_sessions:
        raise RuntimeError(
            "Train과 Val에 중복 세션이 있습니다."
        )

    if train_sessions & test_sessions:
        raise RuntimeError(
            "Train과 Test에 중복 세션이 있습니다."
        )

    if val_sessions & test_sessions:
        raise RuntimeError(
            "Val과 Test에 중복 세션이 있습니다."
        )

    return split_sessions


def find_split(session_name, split_sessions):
    for split, sessions in split_sessions.items():
        if session_name in sessions:
            return split

    return None


def convert_tiff_to_png(
    source_path,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 이미 정상적으로 생성된 파일은 다시 변환하지 않는다.
    if output_path.exists():
        with Image.open(output_path) as image:
            return image.size

    with Image.open(source_path) as image:
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

    clipped = np.clip(
        image_array,
        0,
        MAX_12BIT_VALUE,
    )

    converted = (
        clipped
        / MAX_12BIT_VALUE
        * 255
    ).astype(np.uint8)

    output_image = Image.fromarray(converted)

    # 변환 중 중단될 경우 깨진 최종 파일이 남지 않게 한다.
    temporary_path = output_path.with_suffix(
        ".temporary.png"
    )

    output_image.save(
        temporary_path,
        format="PNG",
    )

    temporary_path.replace(output_path)

    return output_image.size


def convert_label_to_yolo(
    label,
    image_width,
    image_height,
):
    attributes = label.get("attributes", {})

    if attributes.get("relevance") != "relevant":
        return None

    if attributes.get("pictogram") != "circle":
        return None

    state = attributes.get("state")

    if state not in CLASS_IDS:
        return None

    class_id = CLASS_IDS[state]

    x = float(label["x"])
    y = float(label["y"])
    width = float(label["w"])
    height = float(label["h"])

    x1 = max(0.0, min(x, image_width))
    y1 = max(0.0, min(y, image_height))
    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return None

    x_center = (
        ((x1 + x2) / 2.0)
        / image_width
    )

    y_center = (
        ((y1 + y2) / 2.0)
        / image_height
    )

    normalized_width = (
        box_width / image_width
    )

    normalized_height = (
        box_height / image_height
    )

    return {
        "class_id": class_id,
        "state": state,
        "x_center": x_center,
        "y_center": y_center,
        "width": normalized_width,
        "height": normalized_height,
        "original_x": x1,
        "original_y": y1,
        "original_width": box_width,
        "original_height": box_height,
        "track_id": label.get("track_id", ""),
    }


def write_yolo_label(
    output_path,
    yolo_labels,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for label in yolo_labels:
            file.write(
                f"{label['class_id']} "
                f"{label['x_center']:.6f} "
                f"{label['y_center']:.6f} "
                f"{label['width']:.6f} "
                f"{label['height']:.6f}\n"
            )


def write_data_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"

    output_path_text = (
        OUTPUT_ROOT
        .as_posix()
    )

    yaml_text = (
        f"path: {output_path_text}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: red\n"
        f"  1: yellow\n"
        f"  2: green\n"
        f"  3: red_yellow\n"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )


def main():
    print("=" * 80)
    print("DTLD 전체 YOLO 데이터셋 생성 시작")
    print("=" * 80)

    split_sessions = load_split_sessions()

    print(
        "Train 세션 수:",
        len(split_sessions["train"]),
    )
    print(
        "Val 세션 수:",
        len(split_sessions["val"]),
    )
    print(
        "Test 세션 수:",
        len(split_sessions["test"]),
    )

    with JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    images = data["images"]

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_metadata_path = (
        METADATA_DIR
        / "frame_metadata.csv"
    )

    object_metadata_path = (
        METADATA_DIR
        / "object_metadata.csv"
    )

    frame_fields = [
        "split",
        "city",
        "session",
        "sequence",
        "original_image_path",
        "output_image_path",
        "output_label_path",
        "time_stamp",
        "velocity",
        "target_object_count",
    ]

    object_fields = [
        "split",
        "city",
        "session",
        "sequence",
        "original_image_path",
        "output_image_path",
        "time_stamp",
        "velocity",
        "track_id",
        "state",
        "class_id",
        "original_x",
        "original_y",
        "original_width",
        "original_height",
        "x_center",
        "y_center",
        "normalized_width",
        "normalized_height",
    ]

    image_counts = Counter()
    object_counts = Counter()
    empty_label_counts = Counter()

    invalid_box_count = 0
    missing_image_count = 0
    missing_session_count = 0

    with frame_metadata_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as frame_file, object_metadata_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as object_file:

        frame_writer = csv.DictWriter(
            frame_file,
            fieldnames=frame_fields,
        )

        object_writer = csv.DictWriter(
            object_file,
            fieldnames=object_fields,
        )

        frame_writer.writeheader()
        object_writer.writeheader()

        for index, image_info in enumerate(
            images,
            start=1,
        ):
            relative_text = (
                image_info["image_path"]
                .removeprefix("./")
            )

            relative_path = Path(relative_text)

            city = relative_path.parts[0]
            session = relative_path.parts[1]
            time_folder = relative_path.parts[2]

            session_name = (
                f"{city}/{session}"
            )

            sequence_name = (
                f"{city}/{session}/{time_folder}"
            )

            split = find_split(
                session_name,
                split_sessions,
            )

            if split is None:
                missing_session_count += 1
                continue

            source_image_path = (
                DATA_ROOT
                / relative_path
            )

            if not source_image_path.exists():
                missing_image_count += 1
                continue

            output_relative_image = (
                relative_path
                .with_suffix(".png")
            )

            output_relative_label = (
                relative_path
                .with_suffix(".txt")
            )

            output_image_path = (
                OUTPUT_ROOT
                / "images"
                / split
                / output_relative_image
            )

            output_label_path = (
                OUTPUT_ROOT
                / "labels"
                / split
                / output_relative_label
            )

            image_width, image_height = (
                convert_tiff_to_png(
                    source_image_path,
                    output_image_path,
                )
            )

            yolo_labels = []

            for original_label in image_info.get(
                "labels",
                [],
            ):
                converted_label = (
                    convert_label_to_yolo(
                        original_label,
                        image_width,
                        image_height,
                    )
                )

                if converted_label is None:
                    continue

                if (
                    converted_label["width"] <= 0
                    or converted_label["height"] <= 0
                ):
                    invalid_box_count += 1
                    continue

                yolo_labels.append(
                    converted_label
                )

            write_yolo_label(
                output_label_path,
                yolo_labels,
            )

            image_counts[split] += 1

            if not yolo_labels:
                empty_label_counts[split] += 1

            time_stamp = image_info.get(
                "time_stamp",
                "",
            )

            velocity = image_info.get(
                "velocity",
                "",
            )

            relative_output_image_text = (
                output_image_path
                .relative_to(OUTPUT_ROOT)
                .as_posix()
            )

            relative_output_label_text = (
                output_label_path
                .relative_to(OUTPUT_ROOT)
                .as_posix()
            )

            frame_writer.writerow(
                {
                    "split": split,
                    "city": city,
                    "session": session,
                    "sequence": sequence_name,
                    "original_image_path": (
                        str(source_image_path)
                    ),
                    "output_image_path": (
                        relative_output_image_text
                    ),
                    "output_label_path": (
                        relative_output_label_text
                    ),
                    "time_stamp": time_stamp,
                    "velocity": velocity,
                    "target_object_count": (
                        len(yolo_labels)
                    ),
                }
            )

            for label in yolo_labels:
                object_counts[
                    f"{split}_{label['state']}"
                ] += 1

                object_writer.writerow(
                    {
                        "split": split,
                        "city": city,
                        "session": session,
                        "sequence": sequence_name,
                        "original_image_path": (
                            str(source_image_path)
                        ),
                        "output_image_path": (
                            relative_output_image_text
                        ),
                        "time_stamp": time_stamp,
                        "velocity": velocity,
                        "track_id": label["track_id"],
                        "state": label["state"],
                        "class_id": label["class_id"],
                        "original_x": label["original_x"],
                        "original_y": label["original_y"],
                        "original_width": (
                            label["original_width"]
                        ),
                        "original_height": (
                            label["original_height"]
                        ),
                        "x_center": (
                            f"{label['x_center']:.6f}"
                        ),
                        "y_center": (
                            f"{label['y_center']:.6f}"
                        ),
                        "normalized_width": (
                            f"{label['width']:.6f}"
                        ),
                        "normalized_height": (
                            f"{label['height']:.6f}"
                        ),
                    }
                )

            if (
                index % 500 == 0
                or index == len(images)
            ):
                print(
                    f"진행 상황: "
                    f"{index:,} / {len(images):,}"
                )

    write_data_yaml()

    summary = {
        "total_json_images": len(images),
        "image_counts": dict(image_counts),
        "object_counts": dict(object_counts),
        "empty_label_counts": dict(
            empty_label_counts
        ),
        "invalid_box_count": invalid_box_count,
        "missing_image_count": missing_image_count,
        "missing_session_count": missing_session_count,
        "class_names": CLASS_NAMES,
    }

    summary_path = (
        METADATA_DIR
        / "build_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("전체 YOLO 데이터셋 생성 완료")
    print("=" * 80)

    for split in (
        "train",
        "val",
        "test",
    ):
        print()
        print(f"[{split.upper()}]")
        print(
            f"이미지 수: "
            f"{image_counts[split]:,}"
        )
        print(
            f"빈 라벨 이미지 수: "
            f"{empty_label_counts[split]:,}"
        )

        for state in CLASS_IDS:
            count = object_counts[
                f"{split}_{state}"
            ]

            print(
                f"{state} 객체 수: {count:,}"
            )

    print()
    print(
        f"잘못된 박스 수: "
        f"{invalid_box_count}"
    )
    print(
        f"찾을 수 없는 이미지 수: "
        f"{missing_image_count}"
    )
    print(
        f"분할에 없는 세션 이미지 수: "
        f"{missing_session_count}"
    )
    print(f"저장 위치: {OUTPUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
