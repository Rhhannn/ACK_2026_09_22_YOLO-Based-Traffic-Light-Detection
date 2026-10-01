import json
from collections import Counter
from pathlib import Path

from src.temporal_robustness.repository_paths import LABEL_JSON, RAW_DATA_ROOT


DATA_ROOT = RAW_DATA_ROOT
JSON_PATH = LABEL_JSON

def main():
    with JSON_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    images = data["images"]
    session_counts = Counter()

    for image in images:
        parts = Path(image["image_path"]).parts
        city = parts[0]
        session = parts[1]
        session_counts[f"{city}/{session}"] += 1

    print("=" * 60)
    print("세션별 이미지 수")
    print("=" * 60)

    for session_name, count in sorted(session_counts.items()):
        print(f"{session_name:<30} {count:>6}")

    print("=" * 60)
    print(f"전체 주행 세션 수: {len(session_counts)}")
    print(f"JSON 전체 이미지 수: {len(images)}")
    print(f"세션별 이미지 수 합계: {sum(session_counts.values())}")
    print("=" * 60)

    actual_image_paths = {
        path.resolve()
        for path in DATA_ROOT.rglob("*_k0.tiff")
    }

    json_image_paths = {
        (DATA_ROOT / image["image_path"].removeprefix("./")).resolve()
        for image in images
    }

    files_without_json = actual_image_paths - json_image_paths
    json_without_files = json_image_paths - actual_image_paths

    print()
    print("=" * 60)
    print("이미지-JSON 경로 일치 검사")
    print("=" * 60)
    print(f"실제 k0 이미지 수: {len(actual_image_paths)}")
    print(f"JSON 이미지 경로 수: {len(json_image_paths)}")
    print(f"파일은 있지만 JSON에 없는 이미지 수: {len(files_without_json)}")
    print(f"JSON에는 있지만 파일이 없는 이미지 수: {len(json_without_files)}")

    if files_without_json:
        print()
        print("[파일은 있지만 JSON에 없는 이미지]")
        for path in sorted(files_without_json):
            print(path)

    if json_without_files:
        print()
        print("[JSON에는 있지만 파일이 없는 이미지]")
        for path in sorted(json_without_files):
            print(path)

    print("=" * 60)

if __name__ == "__main__":
    main()
