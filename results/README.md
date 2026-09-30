# Results

이 폴더에는 논문의 수치와 통계 검정을 원자료 없이 확인할 수 있는 소형 결과만
포함합니다. DTLD 영상·라벨, 변환 데이터셋, 프레임별 예측 덤프는 포함하지 않습니다.

| 폴더 | 내용 |
|---|---|
| `framewise/` | Clean test의 mAP50 및 mAP50-95 |
| `temporal/` | 조건·모델별 AUC5와 적색 완전 미검출 지속시간 |
| `temporal/track_metrics/` | Track 단위 계산을 확인하기 위한 상세 결과 |
| `temporal/sensitivity/` | confidence·IoU·판정창 민감도 분석 |
| `temporal/sequence_clustered/` | 시퀀스 군집 통계 검정 및 Holm 보정 결과 |
| `training_metadata/` | 모델별 실제 학습 인자와 epoch 기록 |

논문의 각 주장과 파일·열의 정확한 대응은
[`docs/RESULTS_GUIDE.md`](../docs/RESULTS_GUIDE.md)를 참고하십시오.
