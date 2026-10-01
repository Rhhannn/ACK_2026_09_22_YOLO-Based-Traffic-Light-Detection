import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.temporal_robustness.repository_paths import LABEL_JSON, OUTPUTS_ROOT, SPLIT_DIR

JSON_PATH = LABEL_JSON
TEST_SESSION_PATH = SPLIT_DIR / "test_routes.txt"
OUTPUT_PATH = OUTPUTS_ROOT / "statistics" / "test_sequence_statistics.json"


def percentile(values, percent):
    values = sorted(values)

    if not values:
        return 0.0

    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"JSON 파일을 찾을 수 없습니다: {JSON_PATH}"
        )

    if not TEST_SESSION_PATH.exists():
        raise FileNotFoundError(
            f"Test 세션 목록을 찾을 수 없습니다: "
            f"{TEST_SESSION_PATH}"
        )

    with TEST_SESSION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        test_sessions = {
            line.strip()
            for line in file
            if line.strip()
        }

    with JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    sequences = defaultdict(list)
    missing_timestamp_count = 0
    test_image_count = 0

    for image in data["images"]:
        parts = Path(image["image_path"]).parts

        city = parts[0]
        session = parts[1]
        time_folder = parts[2]

        session_name = f"{city}/{session}"

        if session_name not in test_sessions:
            continue

        test_image_count += 1

        timestamp = image.get("time_stamp")

        if timestamp is None:
            missing_timestamp_count += 1
            continue

        sequence_name = (
            f"{city}/{session}/{time_folder}"
        )

        sequences[sequence_name].append(
            float(timestamp)
        )

    frame_counts = []
    durations = []
    frame_intervals = []

    for timestamps in sequences.values():
        timestamps.sort()

        frame_counts.append(len(timestamps))

        if len(timestamps) >= 2:
            durations.append(
                timestamps[-1] - timestamps[0]
            )

        for index in range(1, len(timestamps)):
            interval = (
                timestamps[index]
                - timestamps[index - 1]
            )

            if interval > 0:
                frame_intervals.append(interval)

    sequences_at_least_3 = sum(
        count >= 3
        for count in frame_counts
    )

    sequences_at_least_4 = sum(
        count >= 4
        for count in frame_counts
    )

    sequences_at_least_5 = sum(
        count >= 5
        for count in frame_counts
    )

    result = {
        "test_sessions": sorted(test_sessions),
        "test_session_count": len(test_sessions),
        "test_image_count": test_image_count,
        "sequence_count": len(sequences),
        "missing_timestamp_count": missing_timestamp_count,
        "frame_count": {
            "minimum": min(frame_counts),
            "median": statistics.median(frame_counts),
            "mean": statistics.mean(frame_counts),
            "maximum": max(frame_counts),
        },
        "duration_seconds": {
            "minimum": min(durations),
            "median": statistics.median(durations),
            "mean": statistics.mean(durations),
            "maximum": max(durations),
        },
        "frame_interval_seconds": {
            "minimum": min(frame_intervals),
            "percentile_5": percentile(
                frame_intervals,
                5,
            ),
            "median": statistics.median(
                frame_intervals
            ),
            "mean": statistics.mean(
                frame_intervals
            ),
            "percentile_95": percentile(
                frame_intervals,
                95,
            ),
            "maximum": max(frame_intervals),
        },
        "sequence_length_counts": {
            "at_least_3_frames": sequences_at_least_3,
            "at_least_4_frames": sequences_at_least_4,
            "at_least_5_frames": sequences_at_least_5,
        },
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

    print("=" * 80)
    print("Test 시퀀스 시간 구조 분석 결과")
    print("=" * 80)
    print(f"Test 주행 세션 수: {len(test_sessions)}")
    print(f"Test 이미지 수: {test_image_count}")
    print(f"시간 폴더 시퀀스 수: {len(sequences)}")
    print(
        f"타임스탬프 없는 이미지 수: "
        f"{missing_timestamp_count}"
    )

    print()
    print("[시퀀스별 프레임 수]")
    print(f"최소: {min(frame_counts)}")
    print(
        f"중앙값: "
        f"{statistics.median(frame_counts):.1f}"
    )
    print(
        f"평균: "
        f"{statistics.mean(frame_counts):.1f}"
    )
    print(f"최대: {max(frame_counts)}")

    print()
    print("[시퀀스 지속시간]")
    print(
        f"중앙값: "
        f"{statistics.median(durations):.2f}초"
    )
    print(
        f"평균: "
        f"{statistics.mean(durations):.2f}초"
    )

    print()
    print("[연속 프레임 간격]")
    print(
        f"중앙값: "
        f"{statistics.median(frame_intervals):.3f}초"
    )
    print(
        f"평균: "
        f"{statistics.mean(frame_intervals):.3f}초"
    )
    print(
        f"5%~95% 범위: "
        f"{percentile(frame_intervals, 5):.3f}초"
        f" ~ "
        f"{percentile(frame_intervals, 95):.3f}초"
    )

    print()
    print("[길이별 시퀀스 수]")
    print(f"3프레임 이상: {sequences_at_least_3}")
    print(f"4프레임 이상: {sequences_at_least_4}")
    print(f"5프레임 이상: {sequences_at_least_5}")

    print()
    print(f"결과 저장 위치: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
