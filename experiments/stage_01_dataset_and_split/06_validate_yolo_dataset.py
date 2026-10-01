import json
from collections import Counter
from pathlib import Path

from src.temporal_robustness.repository_paths import DATASET_ROOT, LABEL_JSON

JSON_PATH = LABEL_JSON
YOLO_ROOT = DATASET_ROOT

OUTPUT_PATH = (
    YOLO_ROOT
    / "metadata"
    / "full_validation.json"
)

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

IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1024


def is_target_label(label):
    attributes = label.get("attributes", {})

    return (
        attributes.get("relevance") == "relevant"
        and attributes.get("pictogram") == "circle"
        and attributes.get("state") in CLASS_IDS
    )


def check_original_box(label):
    x = float(label["x"])
    y = float(label["y"])
    width = float(label["w"])
    height = float(label["h"])

    x1 = max(0.0, min(x, IMAGE_WIDTH))
    y1 = max(0.0, min(y, IMAGE_HEIGHT))
    x2 = max(0.0, min(x + width, IMAGE_WIDTH))
    y2 = max(0.0, min(y + height, IMAGE_HEIGHT))

    clipped_width = x2 - x1
    clipped_height = y2 - y1

    return (
        clipped_width > 0
        and clipped_height > 0
    )


def read_yolo_labels(label_path):
    class_counts = Counter()
    errors = []

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        lines = file.readlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{label_path}: "
                f"{line_number}번째 줄의 값이 5개가 아님"
            )
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            errors.append(
                f"{label_path}: "
                f"{line_number}번째 줄을 숫자로 읽을 수 없음"
            )
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"{label_path}: "
                f"잘못된 클래스 번호 {class_id}"
            )
            continue

        values = [
            x_center,
            y_center,
            width,
            height,
        ]

        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):
            errors.append(
                f"{label_path}: "
                f"{line_number}번째 줄이 0~1 범위를 벗어남"
            )
            continue

        if width <= 0 or height <= 0:
            errors.append(
                f"{label_path}: "
                f"{line_number}번째 줄의 박스 크기가 0 이하"
            )
            continue

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        tolerance = 0.00001

        if (
            x1 < -tolerance
            or y1 < -tolerance
            or x2 > 1 + tolerance
            or y2 > 1 + tolerance
        ):
            errors.append(
                f"{label_path}: "
                f"{line_number}번째 박스가 이미지 밖으로 나감"
            )
            continue

        class_counts[
            CLASS_NAMES[class_id]
        ] += 1

    return class_counts, errors


def main():
    print("=" * 80)
    print("전체 YOLO 데이터셋 검사 시작")
    print("=" * 80)

    with JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    expected_counts = Counter()
    convertible_counts = Counter()
    invalid_original_boxes = []

    # 원본 JSON의 학습 대상 객체 수를 다시 계산한다.
    for image_info in data["images"]:
        for label in image_info.get(
            "labels",
            [],
        ):
            if not is_target_label(label):
                continue

            state = label["attributes"]["state"]
            expected_counts[state] += 1

            if check_original_box(label):
                convertible_counts[state] += 1
            else:
                invalid_original_boxes.append(
                    {
                        "image_path": (
                            image_info["image_path"]
                        ),
                        "state": state,
                        "track_id": label.get(
                            "track_id",
                            "",
                        ),
                        "x": label.get("x"),
                        "y": label.get("y"),
                        "w": label.get("w"),
                        "h": label.get("h"),
                    }
                )

    split_results = {}
    all_yolo_counts = Counter()
    all_errors = []
    missing_label_files = []
    missing_image_files = []

    for split in (
        "train",
        "val",
        "test",
    ):
        image_root = (
            YOLO_ROOT
            / "images"
            / split
        )

        label_root = (
            YOLO_ROOT
            / "labels"
            / split
        )

        image_paths = sorted(
            image_root.rglob("*.png")
        )

        label_paths = sorted(
            label_root.rglob("*.txt")
        )

        split_counts = Counter()

        # 모든 PNG에 같은 경로의 TXT가 있는지 검사한다.
        for image_path in image_paths:
            relative_path = image_path.relative_to(
                image_root
            )

            expected_label_path = (
                label_root
                / relative_path.with_suffix(".txt")
            )

            if not expected_label_path.exists():
                missing_label_files.append(
                    str(expected_label_path)
                )

        # 모든 TXT에 같은 경로의 PNG가 있는지 검사한다.
        for number, label_path in enumerate(
            label_paths,
            start=1,
        ):
            relative_path = label_path.relative_to(
                label_root
            )

            expected_image_path = (
                image_root
                / relative_path.with_suffix(".png")
            )

            if not expected_image_path.exists():
                missing_image_files.append(
                    str(expected_image_path)
                )

            counts, errors = read_yolo_labels(
                label_path
            )

            split_counts.update(counts)
            all_yolo_counts.update(counts)
            all_errors.extend(errors)

            if number % 5000 == 0:
                print(
                    f"{split}: "
                    f"라벨 파일 {number:,}개 검사"
                )

        split_results[split] = {
            "image_count": len(image_paths),
            "label_file_count": len(label_paths),
            "class_counts": dict(split_counts),
        }

    result = {
        "json_expected_counts": dict(
            expected_counts
        ),
        "json_convertible_counts": dict(
            convertible_counts
        ),
        "yolo_class_counts": dict(
            all_yolo_counts
        ),
        "invalid_original_box_count": len(
            invalid_original_boxes
        ),
        "invalid_original_boxes": (
            invalid_original_boxes
        ),
        "split_results": split_results,
        "missing_label_file_count": len(
            missing_label_files
        ),
        "missing_image_file_count": len(
            missing_image_files
        ),
        "coordinate_error_count": len(
            all_errors
        ),
        "coordinate_errors": all_errors[:100],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("전체 검사 결과")
    print("=" * 80)

    for state in CLASS_IDS:
        print()
        print(f"[{state}]")
        print(
            f"원본 JSON 객체 수: "
            f"{expected_counts[state]:,}"
        )
        print(
            f"변환 가능한 객체 수: "
            f"{convertible_counts[state]:,}"
        )
        print(
            f"YOLO 객체 수: "
            f"{all_yolo_counts[state]:,}"
        )

    print()

    for split in (
        "train",
        "val",
        "test",
    ):
        split_result = split_results[split]

        print(
            f"{split.upper()}: "
            f"이미지 {split_result['image_count']:,}개 / "
            f"라벨 파일 {split_result['label_file_count']:,}개"
        )

    print()
    print(
        f"원본의 변환 불가능한 박스 수: "
        f"{len(invalid_original_boxes)}"
    )
    print(
        f"누락된 라벨 파일 수: "
        f"{len(missing_label_files)}"
    )
    print(
        f"누락된 이미지 파일 수: "
        f"{len(missing_image_files)}"
    )
    print(
        f"YOLO 좌표 오류 수: "
        f"{len(all_errors)}"
    )

    if invalid_original_boxes:
        print()
        print("[변환 불가능한 원본 박스]")

        for box in invalid_original_boxes:
            print(box)

    print()
    print(f"검사 결과 저장 위치: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
