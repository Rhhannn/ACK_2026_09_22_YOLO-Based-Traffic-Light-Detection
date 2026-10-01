# 패키지 검증 기록

최종 구조 정리 후 2026-10-01에 다음 검증을 수행했습니다.

## 자동 검증 결과

| 검증 | 결과 |
|---|---|
| 전체 Python 파일 AST 문법 검사 | 34개 파일 통과 |
| `tests/test_temporal_metrics_and_degradation.py` 회귀 테스트 | 5개 통과 |
| 로컬 Markdown 링크·이미지 경로 | 49개 확인, 누락 0개 |
| 20번 시퀀스 군집 통계 재계산 | 포함된 CSV/JSON 4개와 SHA-256 완전 일치 |
| 21번 onset curve SVG 재생성 | 포함된 SVG와 SHA-256 완전 일치 |
| 저장소 manifest | Git 표준 내용 96개와 SHA-256 일치 |
| 부호반전 반복 수 | 검정당 100,000회 |

통계 재계산에서 일치한 파일:

- `condition_level_sequence_summary.csv`
- `pairwise_model_sign_flip_tests.csv`
- `degradation_vs_clean_change_sign_flip_tests.csv`
- `sequence_clustered_statistics_full_results.json`

## 실행 명령

### 회귀 테스트

```bash
python -m unittest tests.test_temporal_metrics_and_degradation -v
```

### 원자료 없는 최종 통계 재계산

```powershell
$env:DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT = "outputs\verification_sequence_clustered"
$env:DTLD_CLUSTER_PERMUTATION_ITERATIONS = "100000"
python -m experiments.stage_06_statistical_validation.20_run_sequence_clustered_sign_flip_tests
```

생성된 네 파일의 SHA-256을 `results/05_statistical_validation/`의 동명 파일과
비교했습니다.

### 문법 검사 범위

- `src/`
- `experiments/`
- `tests/`

## 검증 범위의 한계

등록이 필요한 DTLD 원자료와 공개 저장소에서 제외된 모델 가중치가 없으므로, 이번
패키지 정리 단계에서는 영상 변환, 모델 재학습, 전체 추론을 다시 실행하지 않았습니다.
해당 단계의 코드, 실제 분할, 학습 메타데이터와 결과표는 포함되어 있습니다.

가중치 파일을 별도로 받을 경우 [`weights/README.md`](../weights/README.md)의 크기와
SHA-256을 다시 확인해야 합니다.

## Manifest 기준

[`MANIFEST.sha256`](MANIFEST.sha256)은 운영체제별 줄바꿈 변환에 영향을 받지 않도록
커밋된 Git blob의 표준 바이트를 기준으로 생성했습니다. `.gitattributes`에서 텍스트
파일의 줄바꿈을 LF로 고정하므로 새 checkout의 파일 해시도 같은 값이 됩니다.
