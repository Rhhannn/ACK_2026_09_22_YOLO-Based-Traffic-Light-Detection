# 코드 지도

이 문서는 논문의 실험 흐름과 공개 코드의 대응 관계를 보여줍니다. 실행 파일은
`scripts/`에 `01`부터 `21`까지 끊김 없이 배치되어 있으며, 저장소 최상위 폴더에서
`python -m scripts.<모듈명>` 형식으로 실행합니다.

## 전체 파이프라인

| 단계 | 모듈 | 역할 | 주요 출력 |
|---:|---|---|---|
| 01 | `scripts.01_inspect_dtld_dataset` | DTLD 주석 구조와 클래스 수 점검 | 콘솔 점검 결과 |
| 02 | `scripts.02_summarize_routes_and_classes` | 주행 경로별 영상·객체 수 요약 | 경로별 요약 |
| 03 | `scripts.03_recreate_route_disjoint_split` | 결정적 경로 비중복 분할 재생성 | `splits/` |
| 04 | `scripts.04_audit_tiff_pixel_range` | TIFF 비트 깊이와 픽셀 범위 점검 | 콘솔 점검 결과 |
| 05 | `scripts.05_convert_dtld_to_yolo` | DTLD를 4클래스 8-bit YOLO 형식으로 변환 | `data/processed/DTLD_YOLO_4class` |
| 06 | `scripts.06_validate_yolo_dataset` | 변환 영상·라벨·클래스·개수 검증 | 검증 보고 |
| 07 | `scripts.07_train_yolov8n` | 동일 조건으로 YOLOv8n 학습 | `outputs/full_training/yolov8n_final` |
| 08 | `scripts.08_train_yolo11n` | 동일 조건으로 YOLO11n 학습 | `outputs/full_training/yolo11n_final` |
| 09 | `scripts.09_train_yolo12n` | 동일 조건으로 YOLO12n 학습 | `outputs/full_training/yolo12n_final` |
| 10 | `scripts.10_evaluate_clean_models` | Clean test의 프레임 단위 성능 평가 | `outputs/clean_evaluation` |
| 11 | `scripts.11_audit_test_sequences` | Test 시퀀스와 timestamp 구조 점검 | 시퀀스 요약 |
| 12 | `scripts.12_export_clean_predictions` | Clean 프레임 예측 내보내기 | `outputs/temporal_predictions` |
| 13 | `scripts.13_preview_synthetic_degradations` | 합성 열화 예시를 육안 점검 | 미리보기 이미지 |
| 14 | `scripts.14_build_degraded_test_sets` | 9개 고정 합성 열화 test set 생성 | 열화 데이터셋 |
| 15 | `scripts.15_validate_degraded_test_sets` | 열화 영상·복사 라벨·변환 검증 | 검증 보고 |
| 16 | `scripts.16_evaluate_degraded_models` | Clean 및 열화 조건의 프레임 성능 평가 | 열화별 평가 결과 |
| 17 | `scripts.17_export_degraded_predictions` | 시간 분석용 열화 예측 내보내기 | `outputs/degraded_temporal_predictions` |
| 18 | `scripts.18_compute_temporal_metrics` | 논문 정의의 AUC5와 적색 완전 미검출 지속시간 계산 | `outputs/temporal_analysis` |
| 19 | `scripts.19_run_sensitivity_analysis` | confidence·IoU·판정창 민감도 분석 | `outputs/temporal_analysis/sensitivity` |
| 20 | `scripts.20_run_sequence_clustered_statistics` | 시퀀스 군집 부호반전 검정과 Holm 보정 | `outputs/sequence_clustered_statistics` |
| 21 | `scripts.21_generate_onset_curve_figure` | Track 결과에서 논문 곡선 SVG 재생성 | `figures/paper/` |

> **주의:** 03번은 포함된 논문 분할 파일을 다시 씁니다. 논문 수치의 정확한 재현에는
> 저장소에 포함된 `splits/`를 그대로 사용하고, 분할 탐색 자체를 확인할 때만 03번을
> 별도 작업본에서 실행하십시오. 13번은 육안 확인용 선택 단계입니다.

## 공통 코드

| 파일 | 역할 |
|---|---|
| `common/project_paths.py` | 환경변수 기반의 이식 가능한 데이터·출력·가중치 경로 |
| `common/temporal_metrics_core.py` | GT·예측 파싱, Track 구성, IoU 매칭의 공통 구현 |
| `tests/test_core_metrics.py` | 시간 지표와 합성 안개 변환의 회귀 테스트 |

`common/temporal_metrics_core.py`는 실행 단계가 아니라 18번이 불러오는 내부 공통
모듈이므로 번호를 붙이지 않았습니다.

## 설정과 검증 자료

| 경로 | 의미 |
|---|---|
| `configs/paper_experiment.yaml` | 논문에 사용한 학습·열화·시간 지표·통계 설정의 사람이 읽기 쉬운 기록 |
| `configs/dtld_dataset_paths.example.yaml` | 변환 후 생성되는 Ultralytics 데이터 경로 예시 |
| `splits/` | 논문에서 실제 사용한 주행 경로 비중복 분할 |
| `results/` | 논문 수치와 통계 검정을 확인할 수 있는 소형 결과 파일 |
| `results/training_metadata/` | 모델별 학습 인자와 epoch 기록 |
| `weights/README.md` | 공개 가중치 이름, 크기, SHA-256 및 배치 방법 |

`configs/paper_experiment.yaml`은 검토용 설정 기록입니다. 현재 스크립트의 모든 값이
이 YAML에서 자동으로 주입되는 구조는 아니므로, 실행 전에는 YAML과 해당 스크립트의
상수를 함께 확인하십시오. 실제 학습 인자는 `results/training_metadata/<model>/`에도
보존되어 있습니다.

## 관련 문서

- [전체 재현 절차](REPRODUCTION.md)
- [결과 파일 안내](RESULTS_GUIDE.md)
- [패키지 검증 기록](VERIFICATION.md)
- [참고문헌](REFERENCES.md)
