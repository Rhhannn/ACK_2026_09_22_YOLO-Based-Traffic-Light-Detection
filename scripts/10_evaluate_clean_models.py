import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from common.project_paths import DATA_YAML, MODEL_WEIGHTS, OUTPUTS_ROOT

OUTPUT_ROOT = OUTPUTS_ROOT / "clean_evaluation"
MODELS = MODEL_WEIGHTS

CLASS_NAMES = [
    "red",
    "yellow",
    "green",
    "red_yellow",
]


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml을 찾을 수 없습니다: {DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_name} 모델을 찾을 수 없습니다: {model_path}"
            )

    summary_rows = []

    print("=" * 80)
    print("세 모델 최종 Test 평가 시작")
    print("=" * 80)
    print("Test 이미지 수: 4,244")
    print("입력 크기: 1280")
    print("Batch size: 4")
    print("=" * 80)

    for model_name, model_path in MODELS.items():
        print()
        print("=" * 80)
        print(f"{model_name} Test 평가 시작")
        print("=" * 80)

        torch.cuda.empty_cache()

        model = YOLO(str(model_path))

        start_time = time.perf_counter()

        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=1280,
            batch=4,
            device=0,
            workers=2,
            plots=True,
            save_json=True,
            project=str(OUTPUT_ROOT),
            name=f"{model_name}_test",
            exist_ok=True,
        )

        elapsed_minutes = (
            time.perf_counter() - start_time
        ) / 60

        box = metrics.box
        inference_ms = metrics.speed.get(
            "inference",
            0.0,
        )

        summary_rows.append(
            {
                "model": model_name,
                "class": "all",
                "precision": float(box.mp),
                "recall": float(box.mr),
                "mAP50": float(box.map50),
                "mAP50-95": float(box.map),
                "inference_ms": float(inference_ms),
            }
        )

        for class_id, class_name in enumerate(CLASS_NAMES):
            summary_rows.append(
                {
                    "model": model_name,
                    "class": class_name,
                    "precision": float(box.p[class_id]),
                    "recall": float(box.r[class_id]),
                    "mAP50": float(box.ap50[class_id]),
                    "mAP50-95": float(box.ap[class_id]),
                    "inference_ms": float(inference_ms),
                }
            )

        print()
        print(f"{model_name} 평가 완료")
        print(f"소요 시간: {elapsed_minutes:.1f}분")
        print(
            f"추론 시간: {inference_ms:.2f}ms/image"
        )

        del model
        torch.cuda.empty_cache()

    csv_path = OUTPUT_ROOT / "clean_test_framewise_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        fieldnames = [
            "model",
            "class",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "inference_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 80)
    print("세 모델 최종 Test 평가 완료")
    print("=" * 80)

    for row in summary_rows:
        if row["class"] != "all":
            continue

        print(
            f"{row['model']}: "
            f"P={row['precision']:.3f}, "
            f"R={row['recall']:.3f}, "
            f"mAP50={row['mAP50']:.3f}, "
            f"mAP50-95={row['mAP50-95']:.3f}, "
            f"inference={row['inference_ms']:.2f}ms"
        )

    print(f"CSV 저장 위치: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
