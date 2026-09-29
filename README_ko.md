# 합성 열화 조건에서 YOLO 기반 신호등 탐지의 시간적 강건성 평가

[English README](README.md)

이 저장소는 논문 **「합성 열화 조건에서 YOLO 기반 신호등 탐지의 시간적 강건성 평가」**의 재현용 코드 패키지입니다. DTLD(DriveU Traffic Light Dataset)의 원본 영상과 합성 열화 영상에서 YOLOv8n, YOLO11n, YOLO12n을 비교합니다.

저장소에는 논문에서 사용한 분할 목록, 데이터 전처리·학습·평가 코드, 검증 가능한 소형 결과표, 학습 메타데이터만 포함했습니다. DTLD 원자료, 변환 영상, 합성 열화 영상, 프레임별 예측 덤프, 모델 가중치는 포함하지 않았습니다.

## 논문 실험 설정

- 원형 신호등 4개 상태: `red`, `yellow`, `green`, `red_yellow`
- 주행 경로가 겹치지 않는 분할: 학습 34경로, 검증 4경로, 테스트 5경로
- 영상 수: 학습 32,699장, 검증 4,035장, 테스트 4,244장
- 비교 모델: YOLOv8n, YOLO11n, YOLO12n
- 학습: 입력 1280, 배치 4, AdamW, 초기 학습률 0.00125, 가중치 감쇠 0.0005, 시드 42, deterministic 모드, AMP, 최대 50 epoch, 조기 종료 patience 10
- 합성 열화:
  - 안개 alpha: 0.15, 0.30, 0.45
  - 저조도 gamma: 1.4, 2.0, 2.8
  - 수평 모션블러 커널: 3, 7, 11 px
- 기본 시간 지표 설정: confidence 0.25, 매칭 IoU 0.50, 안정 탐지 판정창 2초, 성공률 80%, 최소 관측 3개, 관측 간 최대 간격 2초, AUC 구간 0–5초
- 민감도 분석: confidence 0.10/0.25/0.50, IoU 0.30/0.40/0.50, 판정창 1/2/3초
- 최종 추론 통계: 주행 시퀀스 군집 기준 양측 부호반전 검정 100,000회 및 Holm 보정

시간 지표는 다음 두 가지입니다.

- **AUC5**: 트랙 시작 후 0–5초 동안 누적 안정 탐지 개시 곡선의 정규화 면적입니다. 클수록 좋습니다.
- **평균 최장 연속 완전 미검출 지속시간**(D-bar): 각 트랙에서 적색 신호를 완전히 놓친 가장 긴 연속 구간을 구한 뒤 트랙 간 평균을 낸 값입니다. 작을수록 좋습니다.

## 구성

```text
.
├── configs/                     # 실험 설정과 경로 예시
├── common/                      # 공통 경로 설정
├── data_preparation/            # 데이터 점검·변환·합성 열화
├── figures/                     # 포함 지표에서 재생성한 SVG 그림
├── model_evaluation/            # Clean·열화 평가 및 예측 내보내기
├── model_training/              # YOLOv8n·YOLO11n·YOLO12n 학습
├── results/
│   ├── framewise/               # Clean 테스트 요약
│   ├── temporal/
│   │   ├── track_metrics/       # 보정된 트랙별 시간 지표
│   │   ├── sensitivity/         # confidence·IoU·1/2/3초 민감도 분석
│   │   └── sequence_clustered/  # 논문 최종 시퀀스 군집 통계
│   └── training_metadata/       # 학습 인자와 epoch 기록
├── splits/                      # 논문에서 실제 사용한 경로 분할
├── temporal_analysis/           # AUC5·적색 미검출·민감도·통계
├── tests/                       # 핵심 시간 지표 회귀 테스트
├── visualization/               # 논문 그림 생성 코드
├── weights/README.md            # 가중치 파일명과 SHA-256
├── CODE_STRUCTURE.md            # 전체 코드 위치와 역할
├── CITATION.cff
├── requirements.txt
├── VERIFICATION.md
└── THIRD_PARTY_NOTICES.md
```

`results/`의 CSV·JSON은 DTLD를 대신하는 데이터가 아니라, 최종 통계 계산을 독립적으로 확인하기 위한 소형 결과물입니다.

## DTLD 다운로드와 재배포 주의

DTLD는 Ulm University 공식 페이지에서 직접 신청하고 해당 이용 조건을 따라야 합니다.

- 데이터셋 안내: <https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/>
- 등록 및 이용 조건: <https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/registration-form-dtld/>
- 공식 파서: <https://github.com/julimueller/dtld_parsing>

**DTLD 영상과 라벨은 이 저장소에 올리거나 재배포하면 안 됩니다.** 각 사용자가 데이터 제공기관을 통해 별도로 접근 권한을 받아야 합니다. 자세한 내용은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 확인하십시오.

기본 라벨 경로는 다음과 같습니다.

```text
data/raw/DTLD_Labels_v2.0/v2.0/DTLD_all.json
```

실제 TIFF 파일은 `DTLD_all.json` 안의 상대경로와 일치해야 합니다. 로컬 구성이 다르면 `DTLD_RAW_ROOT`와 `DTLD_LABEL_JSON`을 설정하십시오.

## 설치

별도 Python 환경을 만들고, CUDA 드라이버에 맞는 PyTorch를 먼저 설치한 뒤 다음을 실행합니다.

```bash
python -m pip install -r requirements.txt
```

학습 및 모델 추론 코드는 NVIDIA CUDA GPU를 요구합니다. 필요한 중간 입력이 준비되어 있다면 데이터 점검과 45·47·48번 분석 코드는 CPU에서도 실행할 수 있습니다. 논문 실험에는 Ultralytics 8.4.108을 사용했습니다.

## 로컬 경로 설정

모든 경로는 `common/project_paths.py`에 모았습니다. 기본값은 저장소 내부의 상대경로이며 아래 환경변수로 바꿀 수 있습니다.

| 환경변수 | 기본값 | 용도 |
|---|---|---|
| `DTLD_RAW_ROOT` | `data/raw` | DTLD 원자료 위치 |
| `DTLD_LABEL_JSON` | `data/raw/DTLD_Labels_v2.0/v2.0/DTLD_all.json` | DTLD v2 라벨 JSON |
| `DTLD_DATASET_ROOT` | `data/processed/DTLD_YOLO_4class` | 변환된 YOLO 데이터셋 |
| `DTLD_DEGRADED_ROOT` | `data/processed/DTLD_YOLO_4class_degraded` | 9개 열화 테스트셋 |
| `DTLD_OUTPUT_ROOT` | `outputs` | 학습·추론·분석 출력 |
| `DTLD_WEIGHTS_ROOT` | `weights` | 선택적 공개 가중치 위치 |
| `DTLD_CORRECTED_TEMPORAL_OUTPUT_ROOT` | `outputs/corrected_temporal_analysis` | 45번 출력 |
| `DTLD_CORRECTED_SENSITIVITY_OUTPUT_ROOT` | `outputs/corrected_temporal_analysis/sensitivity` | 47번 출력 |
| `DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT` | `outputs/sequence_clustered_statistics` | 48번 최종 출력 |
| `DTLD_CLUSTER_PERMUTATION_ITERATIONS` | `100000` | 양측 시퀀스 군집 부호반전 횟수 |

PowerShell 예시:

```powershell
$env:DTLD_RAW_ROOT = "<DTLD-경로>"
$env:DTLD_LABEL_JSON = "<DTLD-all-JSON-경로>"
$env:DTLD_OUTPUT_ROOT = "<출력-폴더-경로>"
```

## 전체 재현 순서

명령은 저장소 최상위 폴더에서 실행합니다. 파일 번호가 건너뛰는 부분은 탐색 단계에서만 사용했거나 최종 정의로 대체되어 공개 패키지에서 제외한 코드입니다.

### 1. 데이터 점검과 논문 분할 유지

```bash
python -m data_preparation.01_inspect_dataset
python -m data_preparation.02_session_class_summary
python -m data_preparation.08_check_pixel_range
```

엄밀한 논문 재현에는 이미 들어 있는 `splits/`의 34/4/5 경로 목록을 그대로 사용하십시오. `03_make_split.py`는 분할 탐색 방법을 남기기 위해 포함했지만 분할 파일을 다시 씁니다. 분할 생성 자체를 점검할 때만 별도 복사본에서 실행하는 것이 안전합니다.

### 2. YOLO 데이터셋 변환과 검증

```bash
python -m data_preparation.12_build_full_yolo_dataset
python -m data_preparation.13_validate_full_dataset
```

16-bit TIFF 입력을 8-bit 단일채널 PNG로 변환하고 4개 신호 상태와 경로 분리가 올바른지 검사합니다.

### 3. 모델 학습 또는 공개 가중치 배치

```bash
python -m model_training.19_train_yolov8_final
python -m model_training.20_train_yolo11_final
python -m model_training.21_train_yolo12_final
```

재학습하지 않는 경우 [weights/README.md](weights/README.md)에 적힌 GitHub Release 자산 3개를 내려받아 `weights/`에 둡니다. `.pt` 파일은 Git 저장소 본문에 커밋하지 않습니다.

### 4. Clean 평가와 시간 예측 내보내기

```bash
python -m model_evaluation.22_test_all_models
python -m model_evaluation.23_analyze_test_sequences
python -m model_evaluation.24_export_test_predictions
```

민감도 분석에서 모델을 다시 실행하지 않도록 예측은 confidence 0.01부터 저장합니다.

### 5. 합성 열화 생성·검증·평가

```bash
python -m data_preparation.32_make_degradation_previews
python -m data_preparation.33_build_degraded_test_sets
python -m data_preparation.34_validate_degraded_test_sets
python -m model_evaluation.35_evaluate_degraded_test_sets
python -m model_evaluation.37_export_degraded_temporal_predictions
```

32번은 육안 점검용 미리보기이므로 선택 사항입니다. 33–37번은 9개 열화 테스트셋을 생성·검증하고 프레임 및 시간 분석용 예측을 만듭니다.

### 6. 보정된 시간 지표 계산

```bash
python -m temporal_analysis.45_recalculate_corrected_temporal_metrics
```

45번은 관측 구간의 연속성과 최대 2초 간격을 포함한 논문 최종 정의를 적용합니다. 25번은 45번이 공통 파싱·매칭 함수를 가져오기 때문에 남긴 기반 코드입니다. 25번 안의 과거 탐색용 판정창 목록을 논문 최종 민감도 설정으로 해석하면 안 됩니다.

### 7. 민감도 분석

```bash
python -m temporal_analysis.47_corrected_temporal_sensitivity
```

47번이 논문에서 사용한 **1초·2초·3초 판정창** 및 confidence·IoU 민감도 분석의 기준 코드입니다.

### 8. 논문 최종 시퀀스 군집 통계

```bash
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

48번이 최종 분석입니다. 동일 주행 시퀀스에 속한 트랙을 군집으로 처리하고, 시퀀스 단위 양측 부호반전 검정을 100,000회 수행한 뒤 지표별 직접 비교군과 Clean 대비 변화량 비교군에 각각 Holm 보정을 적용합니다. 별도의 신뢰구간은 산출하지 않습니다. `results/temporal/sequence_clustered/`의 결과가 이 설정으로 생성된 논문 기준 결과입니다.

횟수를 명시적으로 고정하려면 PowerShell에서 다음처럼 실행합니다.

```powershell
$env:DTLD_CLUSTER_PERMUTATION_ITERATIONS = "100000"
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

### 9. 논문 곡선 그림 생성

```bash
python -m visualization.generate_fig4_step_no_ci_svg
```

## 원자료 없이 최종 통계만 검증

보정된 트랙별 CSV를 포함했기 때문에 DTLD와 대용량 예측 JSONL 없이 48번 최종 통계를 다시 실행할 수 있습니다. 기준 결과를 덮어쓰지 않도록 출력을 `outputs/`로 지정하십시오.

```powershell
$env:DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT = "outputs\sequence_clustered_statistics"
$env:DTLD_CLUSTER_PERMUTATION_ITERATIONS = "100000"
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

생성된 CSV·JSON 4개를 `results/temporal/sequence_clustered/`의 기준 파일과 비교하면 됩니다.

## 포함한 파일과 제외한 파일

포함:

- 논문에서 실제 사용한 경로 분할과 분할 요약;
- 전처리, 학습, 평가, 보정 시간 분석, 민감도 분석, 최종 군집 통계에 필요한 코드;
- Clean 결과, 트랙별 시간 지표, 민감도 및 최종 시퀀스 군집 결과표;
- 최종 학습 인자와 epoch 기록;
- 선택적 가중치의 파일명과 SHA-256.

제외:

- DTLD 원본 영상과 라벨 전체;
- 변환한 YOLO 영상·라벨과 9개 합성 열화 영상 세트;
- 프레임별 예측 JSONL과 대용량 평가 출력;
- 캐시, 임시 파일, 미리보기, 로그, 개인 PC 절대경로;
- 중간 체크포인트와 `last.pt`;
- 발표자료, 논문 PDF, 캡처 이미지, 로컬 압축본;
- 대체된 탐색 코드 26–31, 38–44 및 과거 패키징 코드와 보정 전 통계.

## 가중치와 라이선스 상태

세 `best.pt` 파일은 제출 패키지의 `release_assets/model_weights/`에 분리해 두었습니다. 재배포 권한을 확인한 뒤에만 GitHub Release에 첨부하고 [weights/README.md](weights/README.md)의 SHA-256과 대조하십시오.

현재 이 패키지에는 선택된 오픈소스 라이선스가 없으므로 `LICENSE` 파일도 없습니다. 공개 전 저자가 코드 라이선스를 결정하고, 특히 학습 가중치의 재배포 가능 여부를 관련 제3자 조건에 따라 확인해야 합니다. 라이선스가 없다는 사실은 법률상 허용되거나 논문 심사를 위해 명시적으로 허용된 범위를 넘어 코드 사용·재배포 권한을 부여하지 않습니다.

## 인용

최종 서지정보가 확정되면 논문을 인용하십시오. `CITATION.cff`에는 현재 확인된 제목과 저자 정보만 넣었으며 DOI나 저장소 주소를 임의로 만들지 않았습니다.
