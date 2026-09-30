# 전체 재현 절차

[연구 개요](../README.md) · [코드 지도](CODE_MAP.md) ·
[결과 파일 안내](RESULTS_GUIDE.md) · [검증 기록](VERIFICATION.md)

이 문서는 논문 **「합성 열화 조건에서 YOLO 기반 신호등 탐지의 시간적 강건성
평가」**의 데이터 준비부터 최종 통계 검정까지를 재현하는 절차입니다.

저장소에는 논문에서 사용한 분할 목록, 전처리·학습·평가 코드, 검증 가능한 소형
결과표와 학습 메타데이터가 포함됩니다. DTLD 원자료, 변환 영상, 합성 열화 영상,
프레임별 예측 덤프와 모델 가중치는 포함하지 않습니다.

## 논문 실험 설정

- 클래스: `red`, `yellow`, `green`, `red_yellow`
- 주행 경로 비중복 분할: 학습 34개, 검증 4개, 테스트 5개
- 영상 수: 학습 32,699장, 검증 4,035장, 테스트 4,244장
- 모델: YOLOv8n, YOLO11n, YOLO12n
- 학습: 입력 1280, batch 4, AdamW, lr 0.00125, weight decay 0.0005,
  seed 42, deterministic, AMP, 최대 50 epoch, patience 10
- 합성 열화: fog α 0.15/0.30/0.45, low-light γ 1.4/2.0/2.8,
  horizontal motion blur 3/7/11 px
- 기본 시간 지표: confidence 0.25, matching IoU 0.50, 판정창 2초,
  성공률 80%, 최소 관측 3개, 최대 관측 간격 2초, AUC 구간 0–5초
- 최종 통계: 주행 시퀀스 군집 양측 부호반전 검정 100,000회, Holm 보정

설정의 전체 기계 판독본은
[`configs/paper_experiment.yaml`](../configs/paper_experiment.yaml)에 있습니다. 이
파일은 검토용 기록이며 모든 스크립트가 YAML을 자동으로 읽는 구조는 아닙니다.

## 저장소 구성

```text
.
├── common/                      # 경로 및 시간 지표 공통 코드
├── configs/                     # 논문 설정과 데이터 경로 예시
├── docs/                        # 재현·코드·결과·검증 문서
├── figures/
│   ├── readme/                  # README 설명 그림
│   └── paper/                   # 코드로 재생성 가능한 SVG
├── results/                     # 논문 수치와 검증용 소형 결과
├── scripts/                     # 01–21 연속 실행 파이프라인
├── splits/                      # 논문에서 사용한 주행 경로 분할
├── tests/                       # 핵심 계산 회귀 테스트
└── weights/                     # 공개 가중치 배치 안내
```

각 스크립트의 입력·출력은 [코드 지도](CODE_MAP.md)에 정리되어 있습니다.

## 1. DTLD 준비

DTLD는 Ulm University 공식 페이지에서 직접 신청하고 제공기관의 이용 조건을
따라야 합니다.

- [DTLD 안내](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/)
- [등록 및 이용 조건](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/registration-form-dtld/)
- [공식 파서](https://github.com/julimueller/dtld_parsing)

**DTLD 영상과 라벨은 이 저장소에 올리거나 재배포하지 마십시오.** 기본 라벨 경로는
다음과 같습니다.

```text
data/raw/DTLD_Labels_v2.0/v2.0/DTLD_all.json
```

자세한 제3자 자료 안내는
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)를 확인하십시오.

## 2. Python 환경

별도 가상환경을 만들고 CUDA 드라이버에 맞는 PyTorch를 먼저 설치한 뒤 다음을
실행합니다.

```bash
python -m pip install -r requirements.txt
```

논문 실험에는 Ultralytics 8.4.108을 사용했습니다. 학습과 모델 추론에는 NVIDIA
CUDA GPU가 필요합니다. 저장된 Track 결과로 수행하는 20·21번과 회귀 테스트는
CPU 환경에서도 실행할 수 있습니다.

## 3. 로컬 경로 설정

공통 경로는 `common/project_paths.py`에 있습니다. 기본값은 저장소 내부 상대경로이며
환경변수로 바꿀 수 있습니다.

| 환경변수 | 기본값 | 용도 |
|---|---|---|
| `DTLD_RAW_ROOT` | `data/raw` | DTLD 원자료 위치 |
| `DTLD_LABEL_JSON` | `data/raw/DTLD_Labels_v2.0/v2.0/DTLD_all.json` | DTLD v2 라벨 JSON |
| `DTLD_DATASET_ROOT` | `data/processed/DTLD_YOLO_4class` | 변환된 YOLO 데이터셋 |
| `DTLD_DEGRADED_ROOT` | `data/processed/DTLD_YOLO_4class_degraded` | 9개 열화 test set |
| `DTLD_OUTPUT_ROOT` | `outputs` | 학습·추론·분석 출력 |
| `DTLD_WEIGHTS_ROOT` | `weights` | 선택적 공개 가중치 위치 |
| `DTLD_TEMPORAL_OUTPUT_ROOT` | `outputs/temporal_analysis` | 18번 출력 |
| `DTLD_SENSITIVITY_OUTPUT_ROOT` | `outputs/temporal_analysis/sensitivity` | 19번 출력 |
| `DTLD_TEMPORAL_INPUT_ROOT` | `outputs/temporal_analysis` | 20번 Track 입력 |
| `DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT` | `outputs/sequence_clustered_statistics` | 20번 최종 출력 |
| `DTLD_CLUSTER_PERMUTATION_ITERATIONS` | `100000` | 부호반전 반복 횟수 |

PowerShell 예시:

```powershell
$env:DTLD_RAW_ROOT = "<DTLD-경로>"
$env:DTLD_LABEL_JSON = "<DTLD-all-JSON-경로>"
$env:DTLD_OUTPUT_ROOT = "<출력-폴더-경로>"
```

## 4. 전체 실행 순서

모든 명령은 저장소 최상위 폴더에서 실행합니다.

### 4.1 데이터 점검과 분할

```bash
python -m scripts.01_inspect_dtld_dataset
python -m scripts.02_summarize_routes_and_classes
python -m scripts.04_audit_tiff_pixel_range
```

논문 수치의 정확한 재현에는 포함된 `splits/`를 그대로 사용하십시오. 분할 탐색
절차 자체를 확인하려면 별도 작업본에서 다음을 실행합니다. 이 명령은 분할 파일을
다시 씁니다.

```bash
python -m scripts.03_recreate_route_disjoint_split
```

### 4.2 YOLO 데이터셋 변환과 검증

```bash
python -m scripts.05_convert_dtld_to_yolo
python -m scripts.06_validate_yolo_dataset
```

### 4.3 세 모델 학습

```bash
python -m scripts.07_train_yolov8n
python -m scripts.08_train_yolo11n
python -m scripts.09_train_yolo12n
```

재학습하지 않는 경우 [`weights/README.md`](../weights/README.md)에 적힌 가중치를
내려받아 `weights/`에 둡니다. `.pt` 파일은 일반 Git 파일로 커밋하지 않습니다.

### 4.4 Clean 평가와 예측 내보내기

```bash
python -m scripts.10_evaluate_clean_models
python -m scripts.11_audit_test_sequences
python -m scripts.12_export_clean_predictions
```

민감도 분석에서 모델을 다시 실행하지 않도록 예측은 confidence 0.01부터 저장합니다.

### 4.5 합성 열화 생성·검증·평가

```bash
python -m scripts.13_preview_synthetic_degradations
python -m scripts.14_build_degraded_test_sets
python -m scripts.15_validate_degraded_test_sets
python -m scripts.16_evaluate_degraded_models
python -m scripts.17_export_degraded_predictions
```

13번은 육안 점검용 선택 단계입니다. 나머지 단계는 9개 열화 test set을 생성·검증하고
프레임 및 시간 분석용 예측을 만듭니다.

### 4.6 시간 지표와 민감도

```bash
python -m scripts.18_compute_temporal_metrics
python -m scripts.19_run_sensitivity_analysis
```

18번은 연속 관측 구간과 최대 2초 간격을 포함한 논문 최종 정의를 적용합니다.
19번은 confidence, IoU, 1/2/3초 판정창을 한 번에 하나씩 변경합니다.

### 4.7 최종 통계와 그림

```bash
python -m scripts.20_run_sequence_clustered_statistics
python -m scripts.21_generate_onset_curve_figure
```

20번은 동일 주행 시퀀스에 속한 Track을 하나의 군집으로 처리하고, 양측 부호반전
검정을 100,000회 수행한 뒤 지표별 비교군에 Holm 보정을 적용합니다.

## 5. 원자료 없이 핵심 계산 확인

### 회귀 테스트

```bash
python -m unittest tests.test_core_metrics
```

### 최종 통계 재계산

보정된 Track별 CSV가 포함되어 있어 DTLD와 대용량 예측 JSONL 없이 20번 통계를
다시 실행할 수 있습니다. 기본 입력 파일이 없으면 `results/temporal/track_metrics/`
를 자동으로 사용합니다.

```powershell
$env:DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT = "outputs\sequence_clustered_statistics"
$env:DTLD_CLUSTER_PERMUTATION_ITERATIONS = "100000"
python -m scripts.20_run_sequence_clustered_statistics
```

생성된 네 파일을 `results/temporal/sequence_clustered/`의 기준 파일과 비교하십시오.

## 6. 포함·제외 범위

포함:

- 논문에서 실제 사용한 경로 분할과 분할 요약
- 전처리, 학습, 평가, 시간 분석, 민감도 분석, 군집 통계 코드
- Clean 결과, Track별 시간 지표, 민감도 및 최종 통계 결과
- 실제 학습 인자와 epoch 기록
- 선택적 가중치의 파일명과 SHA-256

제외:

- DTLD 원본 영상과 라벨 전체
- 변환한 YOLO 영상·라벨과 9개 합성 열화 영상 세트
- 프레임별 예측 JSONL과 대용량 평가 출력
- 캐시, 임시 파일, 로컬 절대경로, 중간 checkpoint와 `last.pt`
- 발표자료, 논문 PDF, 로컬 압축본

## 7. 라이선스와 인용

현재 공개 코드 라이선스는 선택되지 않았습니다. 이용 가능 범위는
[`LICENSE_STATUS.md`](LICENSE_STATUS.md), 제3자 조건은
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)를 확인하십시오.
서지정보는 [`CITATION.cff`](../CITATION.cff)에 있습니다.
