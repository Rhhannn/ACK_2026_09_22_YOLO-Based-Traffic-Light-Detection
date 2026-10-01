# 결과 파일 안내

이 문서는 논문과 [README](../README.md)의 주요 주장에 대응하는 결과 파일을
안내합니다. 처음 검토할 때는 아래 네 파일만 확인해도 핵심 결과를 추적할 수 있습니다.

## 핵심 결과 파일

| 확인 목적 | 파일 | 확인할 항목 |
|---|---|---|
| Clean 프레임 단위 성능 | [`results/02_clean_baseline/clean_baseline_framewise_metrics.csv`](../results/02_clean_baseline/clean_baseline_framewise_metrics.csv) | `class=all` 행의 `mAP50`, `mAP50-95` |
| 전체 조건의 AUC5와 D̄ | [`results/03_temporal_robustness/auc5_and_red_miss_duration_by_condition.csv`](../results/03_temporal_robustness/auc5_and_red_miss_duration_by_condition.csv) | `condition`, `model`, `auc5`, `mean_longest_complete_red_miss_seconds` |
| 모델 간 통계적 유의성 | [`results/05_statistical_validation/pairwise_model_sign_flip_tests.csv`](../results/05_statistical_validation/pairwise_model_sign_flip_tests.csv) | 효과 차이와 Holm 보정 p-value |
| 실험 설정 | [`configs/paper_experiment.yaml`](../configs/paper_experiment.yaml) | 학습, 열화, 시간 지표 및 통계 검정 설정 |

AUC5는 클수록, D̄는 작을수록 우수합니다. CSV에는 계산 정밀도의 원래 값이
저장되어 있으며 논문과 그림에서는 이를 반올림하여 표시합니다.

## 논문의 주장과 근거 파일

| 논문·README 내용 | 주 근거 | 보조 근거 |
|---|---|---|
| Train/Validation/Test를 주행 경로 단위로 분리 | [`splits/route_disjoint_split_summary.json`](../splits/route_disjoint_split_summary.json) | `splits/*_routes.txt` |
| 세 모델에 동일한 학습 조건 적용 | [`configs/paper_experiment.yaml`](../configs/paper_experiment.yaml) | `results/01_model_training/<model>/training_args.yaml` |
| Clean mAP 차이가 0.004 이내 | [`results/02_clean_baseline/clean_baseline_framewise_metrics.csv`](../results/02_clean_baseline/clean_baseline_framewise_metrics.csv) | 각 모델의 `class=all` 행 |
| Clean AUC5·D̄에서 유의한 모델 차이 없음 | [`results/05_statistical_validation/pairwise_model_sign_flip_tests.csv`](../results/05_statistical_validation/pairwise_model_sign_flip_tests.csv) | `condition=clean` 행 |
| L3 세 조건에서 YOLO11n의 AUC5가 가장 높음 | [`results/03_temporal_robustness/auc5_and_red_miss_duration_by_condition.csv`](../results/03_temporal_robustness/auc5_and_red_miss_duration_by_condition.csv) | `fog_l3`, `lowlight_l3`, `motionblur_l3` 행 |
| YOLO11n 관련 L3 AUC5 비교 6개 중 5개가 유의 | [`results/05_statistical_validation/pairwise_model_sign_flip_tests.csv`](../results/05_statistical_validation/pairwise_model_sign_flip_tests.csv) | `auc5_cluster_holm_p_global` |
| L3 세 조건에서 YOLO11n의 D̄가 수치상 가장 짧음 | [`results/03_temporal_robustness/auc5_and_red_miss_duration_by_condition.csv`](../results/03_temporal_robustness/auc5_and_red_miss_duration_by_condition.csv) | `mean_longest_complete_red_miss_seconds` |
| Fog·Motion blur의 D̄ 차이는 유의하고 Low-light는 비유의 | [`results/05_statistical_validation/pairwise_model_sign_flip_tests.csv`](../results/05_statistical_validation/pairwise_model_sign_flip_tests.csv) | `red_duration_cluster_holm_p_global` |
| L3 안정 탐지 개시 곡선 | [`results/03_temporal_robustness/track_level/stable_detection_onset_by_track.csv`](../results/03_temporal_robustness/track_level/stable_detection_onset_by_track.csv) | Track별 `stable_start_seconds` |
| AUC5 민감도 분석에서 YOLO11n이 21/21 설정 1위 | [`results/04_sensitivity_analysis/best_model_by_sensitivity_setting.csv`](../results/04_sensitivity_analysis/best_model_by_sensitivity_setting.csv) | 중복 기본 설정을 제외한 고유 평가 설정 |
| D̄ 민감도 분석에서 YOLO11n이 8/9 설정 1위 | [`results/04_sensitivity_analysis/best_model_consistency_by_parameter_family.csv`](../results/04_sensitivity_analysis/best_model_consistency_by_parameter_family.csv) | `metric=red_duration` |
| Clean 대비 열화 성능 변화량 비교 | [`results/05_statistical_validation/degradation_vs_clean_change_sign_flip_tests.csv`](../results/05_statistical_validation/degradation_vs_clean_change_sign_flip_tests.csv) | 차이의 차이와 Holm 보정 p-value |

## 통계 결과 해석

`pairwise_model_sign_flip_tests.csv`의 `comparison`은 왼쪽 모델에서 오른쪽 모델을
뺀 비교입니다.

- `auc5_difference_points > 0`: 왼쪽 모델의 AUC5가 더 높음
- `red_duration_difference_seconds < 0`: 왼쪽 모델의 D̄가 더 짧음
- `*_holm_p_global < 0.05`: 해당 지표의 직접 모델 비교군에서 Holm 보정 후 유의

`degradation_vs_clean_change_sign_flip_tests.csv`는 L3 조건의 직접 성능이 아니라, 각 모델의
Clean 대비 변화량을 비교한 보조 분석입니다.

## 결과 파일의 역할

### Primary results

- `results/02_clean_baseline/clean_baseline_framewise_metrics.csv`
- `results/03_temporal_robustness/auc5_and_red_miss_duration_by_condition.csv`
- `results/05_statistical_validation/pairwise_model_sign_flip_tests.csv`

논문과 README의 핵심 표, 수치 및 통계적 결론에 직접 사용된 파일입니다.

### Robustness checks

- `results/04_sensitivity_analysis/`
- `results/05_statistical_validation/degradation_vs_clean_change_sign_flip_tests.csv`

평가 기준 변경과 Clean 대비 변화량을 이용해 주 결론의 안정성을 점검한 결과입니다.

### Audit-level outputs

- `results/03_temporal_robustness/track_level/`
- `results/01_model_training/`
- `results/04_sensitivity_analysis/*.json` 및 `results/05_statistical_validation/*.json`

Track 단위 계산 검증, 학습 이력 확인 및 전체 결과의 기계 판독을 위한 상세
출력입니다. JSON 파일은 여러 CSV 결과를 하나의 구조로 보관한 전체 기록이며 별도의
추가 실험 결과가 아닙니다.

## 민감도 분석 행 수 주의

AUC5 민감도 파일에는 confidence, IoU, window 분석별로 기본 설정이 반복되어
있습니다. 표의 행을 단순 합산하면 27개로 보이지만,
`(condition, confidence, iou, window_seconds)` 조합을 기준으로 중복을 제거하면
논문에 보고된 고유 설정은 21개입니다.

## 데이터 분할

- 학습: `splits/train_routes.txt`
- 검증: `splits/validation_routes.txt`
- 테스트: `splits/test_routes.txt`
- 분할별 영상·객체 수와 중복 여부: `splits/route_disjoint_split_summary.json`
