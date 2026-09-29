import time
from pathlib import Path

import torch
from ultralytics import YOLO

from common.project_paths import DATA_YAML, OUTPUTS_ROOT

OUTPUT_ROOT = OUTPUTS_ROOT / "full_training"


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml을 찾을 수 없습니다: {DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다."
        )

    print("=" * 80)
    print("YOLOv8n 최종 학습 시작")
    print("=" * 80)
    print(f"데이터: {DATA_YAML}")
    print("모델: yolov8n.pt")
    print("최대 Epoch: 50")
    print("Early stopping patience: 10")
    print("입력 크기: 1280")
    print("Batch size: 4")
    print("GPU: NVIDIA GPU 0")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = YOLO("yolov8n.pt")

    start_time = time.perf_counter()

    model.train(
        data=str(DATA_YAML),

        # 공통 학습 조건
        epochs=50,
        patience=10,
        imgsz=1280,
        batch=4,
        device=0,
        workers=2,

        # 공통 최적화 조건
        optimizer="AdamW",
        lr0=0.00125,
        weight_decay=0.0005,

        # 재현성 설정
        seed=42,
        deterministic=True,

        # GPU 및 데이터 설정
        amp=True,
        cache=False,

        # 검증 및 결과 저장
        val=True,
        plots=True,
        save=True,
        save_period=5,

        project=str(OUTPUT_ROOT),
        name="yolov8n_final",
        exist_ok=False,
    )

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_hours = elapsed_seconds / 3600

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    result_folder = OUTPUT_ROOT / "yolov8n_final"

    print()
    print("=" * 80)
    print("YOLOv8n 최종 학습 완료")
    print("=" * 80)
    print(f"전체 소요 시간: {elapsed_hours:.2f}시간")
    print(f"최대 GPU 메모리: {peak_memory_gb:.2f}GB")
    print(f"전체 결과: {result_folder}")
    print(f"최적 모델: {result_folder / 'weights' / 'best.pt'}")
    print(f"마지막 모델: {result_folder / 'weights' / 'last.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
