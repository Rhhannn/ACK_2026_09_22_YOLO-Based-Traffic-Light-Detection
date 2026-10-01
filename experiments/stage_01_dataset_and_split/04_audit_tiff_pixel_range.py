import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.temporal_robustness.repository_paths import LABEL_JSON, OUTPUTS_ROOT, RAW_DATA_ROOT

DATA_ROOT = RAW_DATA_ROOT
JSON_PATH = LABEL_JSON
OUTPUT_DIR = OUTPUTS_ROOT / "statistics"

OUTPUT_PATH = OUTPUT_DIR / "pixel_range_sample.json"

# 전체 이미지 중 골고루 뽑아서 검사할 개수
SAMPLE_COUNT = 500


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DTLD TIFF 픽셀 범위 검사 시작")
    print("=" * 70)

    with JSON_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    images = data["images"]
    total_image_count = len(images)

    # 전체 데이터의 앞·중간·뒤에서 골고루 500장을 선택한다.
    sample_indices = np.linspace(
        0,
        total_image_count - 1,
        num=min(SAMPLE_COUNT, total_image_count),
        dtype=int,
    )

    sample_indices = sorted(set(sample_indices.tolist()))

    minimum_values = []
    maximum_values = []

    image_modes = Counter()
    image_sizes = Counter()

    missing_files = []
    over_12bit_files = []

    for number, image_index in enumerate(sample_indices, start=1):
        image_info = images[image_index]

        relative_path = image_info["image_path"].removeprefix("./")
        image_path = DATA_ROOT / relative_path

        if not image_path.exists():
            missing_files.append(str(image_path))
            continue

        with Image.open(image_path) as image:
            image_modes[image.mode] += 1
            image_sizes[f"{image.width}x{image.height}"] += 1

            image_array = np.asarray(image)

        minimum = int(image_array.min())
        maximum = int(image_array.max())

        minimum_values.append(minimum)
        maximum_values.append(maximum)

        # 12비트의 최대값은 4095다.
        if maximum > 4095:
            over_12bit_files.append(
                {
                    "path": str(image_path),
                    "maximum": maximum,
                }
            )

        if number % 50 == 0 or number == len(sample_indices):
            print(
                f"검사 진행: {number} / {len(sample_indices)}"
            )

    if not maximum_values:
        print("오류: 정상적으로 읽은 이미지가 없습니다.")
        return

    result = {
        "total_json_images": total_image_count,
        "sampled_images": len(sample_indices),
        "successfully_read_images": len(maximum_values),
        "missing_file_count": len(missing_files),
        "image_modes": dict(image_modes),
        "image_sizes": dict(image_sizes),
        "smallest_pixel_value": min(minimum_values),
        "largest_pixel_value": max(maximum_values),
        "median_image_maximum": float(
            np.median(maximum_values)
        ),
        "percentile_95_image_maximum": float(
            np.percentile(maximum_values, 95)
        ),
        "images_over_4095": len(over_12bit_files),
        "missing_files": missing_files,
        "over_12bit_files": over_12bit_files,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("검사 완료")
    print("=" * 70)
    print(f"JSON 전체 이미지 수: {total_image_count}")
    print(f"검사 대상으로 선택한 이미지 수: {len(sample_indices)}")
    print(f"정상적으로 읽은 이미지 수: {len(maximum_values)}")
    print(f"찾을 수 없는 이미지 수: {len(missing_files)}")
    print(f"이미지 모드: {dict(image_modes)}")
    print(f"이미지 크기: {dict(image_sizes)}")
    print(f"전체 표본의 가장 작은 픽셀값: {min(minimum_values)}")
    print(f"전체 표본의 가장 큰 픽셀값: {max(maximum_values)}")
    print(
        "이미지별 최댓값의 중앙값: "
        f"{np.median(maximum_values):.1f}"
    )
    print(
        "이미지별 최댓값의 상위 95% 지점: "
        f"{np.percentile(maximum_values, 95):.1f}"
    )
    print(f"4095를 넘는 이미지 수: {len(over_12bit_files)}")
    print(f"검사 결과 저장 위치: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
