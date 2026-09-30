import csv
import json
from collections import defaultdict
from pathlib import Path

from common.project_paths import LABEL_JSON, OUTPUTS_ROOT

JSON_PATH = LABEL_JSON
OUTPUT_DIR = OUTPUTS_ROOT / "statistics"
OUTPUT_CSV = OUTPUT_DIR / "session_class_summary.csv"

TARGET_STATES = ("red", "yellow", "green", "red_yellow")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    summary = defaultdict(
        lambda: {
            "image_count": 0,
            "red_objects": 0,
            "yellow_objects": 0,
            "green_objects": 0,
            "red_yellow_objects": 0,
            "red_images": 0,
            "yellow_images": 0,
            "green_images": 0,
            "red_yellow_images": 0,
        }
    )

    for image in data["images"]:
        parts = Path(image["image_path"]).parts
        city = parts[0]
        session = parts[1]
        session_name = f"{city}/{session}"

        summary[session_name]["image_count"] += 1

        states_in_image = set()

        for label in image["labels"]:
            attributes = label["attributes"]

            if attributes["relevance"] != "relevant":
                continue

            if attributes["pictogram"] != "circle":
                continue

            state = attributes["state"]

            if state not in TARGET_STATES:
                continue

            summary[session_name][f"{state}_objects"] += 1
            states_in_image.add(state)

        for state in states_in_image:
            summary[session_name][f"{state}_images"] += 1

    fieldnames = [
        "session",
        "image_count",
        "red_objects",
        "yellow_objects",
        "green_objects",
        "red_yellow_objects",
        "red_images",
        "yellow_images",
        "green_images",
        "red_yellow_images",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for session_name in sorted(summary):
            row = {"session": session_name}
            row.update(summary[session_name])
            writer.writerow(row)

    print("=" * 80)
    print("세션별 클래스 분포")
    print("=" * 80)

    for session_name in sorted(summary):
        row = summary[session_name]

        print(
            f"{session_name:<28} "
            f"이미지={row['image_count']:>4} | "
            f"red={row['red_objects']:>5} | "
            f"yellow={row['yellow_objects']:>4} | "
            f"green={row['green_objects']:>5} | "
            f"red_yellow={row['red_yellow_objects']:>4}"
        )

    print("=" * 80)
    print(f"CSV 저장 완료: {OUTPUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
