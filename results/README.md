# Results

이 폴더에는 논문의 수치와 통계 검정을 원자료 없이 확인할 수 있는 소형 결과만
포함합니다. 폴더 번호는 **학습 기록 → Clean 기준선 → 시간적 강건성 → 민감도 분석
→ 통계 검정**의 연구 흐름을 나타냅니다.

| 결과 단계 | 내용 | 대응 실험 |
|---|---|---:|
| `01_model_training/` | 모델별 실제 학습 인자와 epoch 기록 | 07–09 |
| `02_clean_baseline/` | Clean test의 mAP50 및 mAP50-95 | 10 |
| `03_temporal_robustness/` | 조건·모델별 AUC5와 적색 완전 미검출 지속시간 | 18 |
| `03_temporal_robustness/track_level/` | Track 단위 시간 지표 계산을 확인하는 상세 결과 | 18 |
| `04_sensitivity_analysis/` | confidence·IoU·판정창 변화에 따른 순위 일관성 | 19 |
| `05_statistical_validation/` | 시퀀스 군집 부호반전 검정과 Holm 보정 결과 | 20 |

핵심 파일명도 내용을 직접 드러내도록 구성했습니다.

- `clean_baseline_framewise_metrics.csv`
- `auc5_and_red_miss_duration_by_condition.csv`
- `stable_detection_onset_by_track.csv`
- `red_miss_duration_by_track.csv`
- `pairwise_model_sign_flip_tests.csv`

DTLD 영상·라벨, 변환 데이터셋, 프레임별 예측 덤프와 모델 가중치는 포함하지
않습니다. 논문의 각 주장과 파일·열의 정확한 대응은
[`docs/RESULTS_GUIDE.md`](../docs/RESULTS_GUIDE.md)를 참고하십시오.
