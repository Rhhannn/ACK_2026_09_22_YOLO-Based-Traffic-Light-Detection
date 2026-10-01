import csv
import json
import random
from pathlib import Path

from src.temporal_robustness.repository_paths import OUTPUTS_ROOT, SPLIT_DIR

INPUT_CSV = (
    OUTPUTS_ROOT
    / "statistics"
    / "session_class_summary.csv"
)

OUTPUT_DIR = SPLIT_DIR

TARGET_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

FEATURES = [
    "image_count",
    "red_objects",
    "yellow_objects",
    "green_objects",
    "red_yellow_objects",
]

RANDOM_SEED = 42
SEARCH_ITERATIONS = 200000


def load_sessions():
    sessions = []

    with INPUT_CSV.open("r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            converted = {"session": row["session"]}

            for feature in FEATURES:
                converted[feature] = int(row[feature])

            sessions.append(converted)

    return sessions


def calculate_totals(sessions):
    return {
        feature: sum(session[feature] for session in sessions)
        for feature in FEATURES
    }


def calculate_split_totals(assignments):
    split_totals = {
        split: {feature: 0 for feature in FEATURES}
        for split in TARGET_RATIOS
    }

    for split, sessions in assignments.items():
        for session in sessions:
            for feature in FEATURES:
                split_totals[split][feature] += session[feature]

    return split_totals


def calculate_score(assignments, overall_totals):
    split_totals = calculate_split_totals(assignments)

    score = 0.0

    # 이미지 수와 클래스 객체 수 비율 오차
    for split, target_ratio in TARGET_RATIOS.items():
        for feature in FEATURES:
            total = overall_totals[feature]

            if total == 0:
                continue

            actual_ratio = split_totals[split][feature] / total
            difference = actual_ratio - target_ratio

            # 희소 클래스에 더 높은 중요도
            if feature == "red_yellow_objects":
                weight = 3.0
            elif feature == "yellow_objects":
                weight = 2.0
            elif feature == "red_objects":
                weight = 1.2
            else:
                weight = 1.0

            score += weight * (difference ** 2)

    # Val과 Test에 모든 클래스가 반드시 포함되도록 강한 벌점
    for split in ["val", "test"]:
        for feature in [
            "red_objects",
            "yellow_objects",
            "green_objects",
            "red_yellow_objects",
        ]:
            if split_totals[split][feature] == 0:
                score += 1000

    # 각 분할에 최소 1개 세션이 존재하도록 검사
    for split in TARGET_RATIOS:
        if len(assignments[split]) == 0:
            score += 1000

    return score


def make_random_assignment(sessions, random_generator):
    shuffled = sessions.copy()
    random_generator.shuffle(shuffled)

    total_sessions = len(shuffled)

    # 43개 기준 대략 Train 35, Val 4, Test 4
    train_count = round(total_sessions * TARGET_RATIOS["train"])
    val_count = round(total_sessions * TARGET_RATIOS["val"])

    train_sessions = shuffled[:train_count]
    val_sessions = shuffled[train_count : train_count + val_count]
    test_sessions = shuffled[train_count + val_count :]

    return {
        "train": train_sessions,
        "val": val_sessions,
        "test": test_sessions,
    }


def save_results(assignments, overall_totals):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_totals = calculate_split_totals(assignments)

    # 경로 목록 저장
    output_names = {
        "train": "train_routes.txt",
        "val": "validation_routes.txt",
        "test": "test_routes.txt",
    }
    for split, sessions in assignments.items():
        output_txt = OUTPUT_DIR / output_names[split]

        with output_txt.open("w", encoding="utf-8") as file:
            for session in sorted(sessions, key=lambda item: item["session"]):
                file.write(session["session"] + "\n")

    # 전체 분할 정보 JSON 저장
    output_json = OUTPUT_DIR / "route_disjoint_split_summary.json"

    result = {
        "target_ratios": TARGET_RATIOS,
        "overall_totals": overall_totals,
        "splits": {},
    }

    for split, sessions in assignments.items():
        result["splits"][split] = {
            "session_count": len(sessions),
            "sessions": sorted(
                session["session"]
                for session in sessions
            ),
            "totals": split_totals[split],
            "ratios": {
                feature: (
                    split_totals[split][feature]
                    / overall_totals[feature]
                    if overall_totals[feature] > 0
                    else 0
                )
                for feature in FEATURES
            },
        }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def print_results(assignments, overall_totals, best_score):
    split_totals = calculate_split_totals(assignments)

    print("=" * 100)
    print("최종 Train / Val / Test 세션 분할")
    print("=" * 100)
    print(f"최적화 점수: {best_score:.10f}")
    print()

    for split in ["train", "val", "test"]:
        sessions = assignments[split]
        totals = split_totals[split]

        print(f"[{split.upper()}]")
        print(f"세션 수: {len(sessions)}")
        print("세션 목록:")

        for session in sorted(sessions, key=lambda item: item["session"]):
            print(f"  - {session['session']}")

        print()
        print(
            f"이미지: {totals['image_count']} "
            f"({totals['image_count'] / overall_totals['image_count'] * 100:.2f}%)"
        )
        print(
            f"red: {totals['red_objects']} "
            f"({totals['red_objects'] / overall_totals['red_objects'] * 100:.2f}%)"
        )
        print(
            f"yellow: {totals['yellow_objects']} "
            f"({totals['yellow_objects'] / overall_totals['yellow_objects'] * 100:.2f}%)"
        )
        print(
            f"green: {totals['green_objects']} "
            f"({totals['green_objects'] / overall_totals['green_objects'] * 100:.2f}%)"
        )
        print(
            f"red_yellow: {totals['red_yellow_objects']} "
            f"({totals['red_yellow_objects'] / overall_totals['red_yellow_objects'] * 100:.2f}%)"
        )
        print("-" * 100)


def main():
    sessions = load_sessions()
    overall_totals = calculate_totals(sessions)

    random_generator = random.Random(RANDOM_SEED)

    best_assignment = None
    best_score = float("inf")

    for iteration in range(SEARCH_ITERATIONS):
        assignment = make_random_assignment(
            sessions,
            random_generator,
        )

        score = calculate_score(
            assignment,
            overall_totals,
        )

        if score < best_score:
            best_score = score
            best_assignment = assignment

        if (iteration + 1) % 20000 == 0:
            print(
                f"탐색 진행: {iteration + 1:,} / "
                f"{SEARCH_ITERATIONS:,}, "
                f"현재 최적 점수: {best_score:.10f}"
            )

    print_results(
        best_assignment,
        overall_totals,
        best_score,
    )

    save_results(
        best_assignment,
        overall_totals,
    )

    print()
    print("=" * 100)
    print("분할 결과 저장 완료")
    print(f"저장 위치: {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
