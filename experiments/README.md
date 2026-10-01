# Experiment pipeline

이 폴더는 논문의 실험을 **연구 수행 순서대로 7개 단계**로 구분합니다. 실행 파일은
전체 흐름에서의 순서를 나타내는 `01`–`21` 번호를 유지하므로, 폴더를 펼치지 않아도
실험의 선후관계를 확인할 수 있습니다.

```text
DTLD 점검·분할 → 모델 학습 → Clean 기준선 → 합성 열화 → 시간적 강건성
               → 통계 검정 → 논문 그림
```

| 연구 단계 | 실행 번호 | 목적 |
|---|---:|---|
| `stage_01_dataset_and_split/` | 01–06 | DTLD 점검, 경로 비중복 분할, YOLO 변환·검증 |
| `stage_02_model_training/` | 07–09 | YOLOv8n·YOLO11n·YOLO12n 동일 조건 학습 |
| `stage_03_clean_baseline/` | 10–12 | Clean 성능과 시퀀스 구조 점검, 예측 내보내기 |
| `stage_04_synthetic_degradation/` | 13–17 | Fog·Low-light·Motion blur 생성, 검증, 평가 |
| `stage_05_temporal_robustness/` | 18–19 | AUC5·적색 완전 미검출 지속시간 및 민감도 분석 |
| `stage_06_statistical_validation/` | 20 | 시퀀스 군집 부호반전 검정과 Holm 보정 |
| `stage_07_figures/` | 21 | Track 결과에서 논문 그림 재생성 |

## 파일명 규칙

실행 파일은 `<전체 순번>_<동작>_<대상>.py` 형식을 사용합니다. 예를 들어
`18_compute_auc5_and_red_miss_duration.py`는 전체 파이프라인의 18번째 단계이며,
두 시간적 강건성 지표를 계산하는 파일임을 이름만으로 알 수 있습니다.

## 실행 방법

모든 명령은 저장소 최상위 폴더에서 실행합니다.

```bash
python -m experiments.stage_05_temporal_robustness.18_compute_auc5_and_red_miss_duration
```

전체 명령과 입력·출력은 [재현 절차](../docs/REPRODUCTION.md), 논문 단계와 코드의
대응은 [코드 지도](../docs/CODE_MAP.md)를 참고하십시오. 번호가 없는 재사용 코드는
[`src/temporal_robustness/`](../src/temporal_robustness/)에 분리되어 있습니다.
