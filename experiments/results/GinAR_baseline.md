# GinAR Baseline 실험 결과

## 개요
- **모델**: GinAR (Yu et al., KDD 2024) — end-to-end VSF 모델
- **코드**: `/home/elicer/VSF_Unified/external/ginar/` 원본 모델 수정 없이 사용
- **실험 코드**: `experiments/ginar_baseline/run_ginar.py`
- **날짜**: 2026-03-19

## 실험 조건

### 우리 실험 vs GinAR 논문 조건 비교

| 항목 | GinAR 논문 | 우리 실험 | 차이 |
|------|-----------|----------|------|
| in_size | 3 (값 + time-of-day + day-of-week) | 1 (값만) | 시간 feature 제거 |
| missing rate | 90% (10% 관측) | 85% (15% 관측) | 관측 비율 다름 |
| data split | 60/20/20 | 70/10/20 | COMET/VIDA 프로토콜 따름 |
| 마스킹 방식 | 고정 마스크 (모든 샘플 동일) | 랜덤 마스크 per batch | COMET 프로토콜 따름 |
| 평가 지표 | 전체 변수 MAE/RMSE | ObsMAE/ObsRMSE (관측 변수만) | COMET/VIDA 프로토콜 따름 |
| 평가 방식 | 전체 test 한번 | 100-mask 반복 평가 | COMET/VIDA 프로토콜 따름 |
| RMSE 계산 | - | per-horizon 평균 | Chauhan/VIDA 호환 |
| adj matrix | pre-computed (doubletransition) | identity (Solar/ECG), doubletransition (METR-LA) | 데이터셋별 상이 |
| 데이터셋 | METR-LA, PEMS-BAY, PEMS04, PEMS08, AQI | Solar, ECG5000, METR-LA | 겹치는건 METR-LA뿐 |
| 정규화 | min-max | min-max | 동일 |
| seed | 1 | 1 (seed=0) | 동일 |
| epochs | 100 | 100 (early stopping patience=15) | 동일 |
| LR | 0.006 | 0.006 | 동일 |
| batch_size | 16 | 16 | 동일 |
| emb_size | 16 | 16 | 동일 |
| grap_size | 8 | 8 | 동일 |
| layer_num | 2 | 2 | 동일 |

### 조건 변경 이유

- **in_size=1**: COMET/VIDA/Chauhan 모두 시간 feature를 사용하지 않음. 공정한 비교를 위해 통일. GinAR에 불리할 수 있음.
- **missing_rate=85%**: COMET/VIDA/Chauhan의 표준 프로토콜 (15% 관측). GinAR 논문은 90%.
- **data_split=70/10/20**: COMET/VIDA/Chauhan 표준. GinAR 논문은 60/20/20.
- **랜덤 마스킹**: COMET은 매 배치 랜덤 마스킹으로 학습. GinAR 원본은 고정 마스크. 공정성을 위해 랜덤 마스킹 적용.
- **ObsMAE**: COMET/VIDA 프로토콜. GinAR 원본은 전체 변수 MAE.

## 결과 (seed 0, 100-mask)

| Dataset | N | GinAR ObsMAE | GinAR ObsRMSE | best_epoch | params |
|---------|---|-------------|---------------|------------|--------|
| ECG5000 | 140 | 6.32 ± 1.43 | 9.92 ± 2.56 | 23 | 63,312 |
| Solar | 137 | 7.54 ± 0.55 | 9.25 ± 0.89 | 10 | 61,236 |
| METR-LA | 207 | 6.83 ± 0.74 | 13.20 ± 1.13 | - | 119,056 |

## COMET/VIDA 비교

| Dataset | GinAR | COMET | VIDA | COMET vs GinAR |
|---------|-------|-------|------|---------------|
| ECG5000 | 6.32(1.43) | **3.12(0.54)** | 3.22(0.58) | -50.6% |
| Solar | 7.54(0.55) | **2.10(0.18)** | 2.33(0.25) | -72.1% |
| METR-LA | 6.83(0.74) | TBD | **3.36(0.20)** | - |

## GinAR 논문 원본 결과 (참고, 직접 비교 불가)

GinAR 논문 Table 2 (METR-LA, missing_rate=90%, in_size=3):
- MAE: 3.87, RMSE: 7.84, MAPE: 11.25

우리 실험과 직접 비교 불가능한 이유:
- in_size 다름 (3 vs 1)
- missing_rate 다름 (90% vs 85%)
- data_split 다름 (60/20/20 vs 70/10/20)
- 평가 지표 다름 (전체 MAE vs ObsMAE)

## 분석

### GinAR 성능이 낮은 이유

1. **in_size=1 (시간 feature 없음)**: GinAR 논문은 time-of-day, day-of-week feature를 사용. 교통/에너지 데이터는 시간대별 패턴이 크므로 시간 feature 제거 시 성능 하락이 큼.
2. **파라미터 수 (61~119K)**: COMET(1.48M), VIDA(6.3M) 대비 매우 작음. 모델 용량 자체가 부족.
3. **identity adj**: Solar, ECG5000에서 그래프 정보 없이 학습. GinAR은 그래프 기반 모델이라 adj가 중요.
4. **랜덤 마스킹**: GinAR 원본은 고정 마스크로 학습하여 특정 결측 패턴에 최적화. 랜덤 마스킹은 더 어려운 조건.

### 논문에서의 활용

- GinAR은 유일한 e2e baseline으로, codebook/gating 없는 e2e의 한계를 보여줌
- 조건 차이(in_size, missing_rate 등)를 논문에 명시해야 함
- "동일 조건(in_size=1, 85% missing, ObsMAE)에서 비교"임을 명확히 서술

## 파일 경로

- 데이터 준비: `experiments/ginar_baseline/prepare_data.py`
- 실행 코드: `experiments/ginar_baseline/run_ginar.py`
- 데이터: `experiments/ginar_baseline/data/{ecg5000,solar,metr-la}/`
- 결과: `experiments/ginar_baseline/logs/ginar_{dataset}_s0/results.json`
- METR-LA adj: `data/raw/adj_mx_metrla.pkl` → `experiments/ginar_baseline/data/metr-la/adj_metr-la.pkl`로 복사
