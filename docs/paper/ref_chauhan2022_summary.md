# VIDA (VSF 원본 논문) 분석 요약

## 서지정보
- **제목:** Multi-Variate Time Series Forecasting on Variable Subsets
- **저자:** Jatin Chauhan, Aravindan Raghuveer, Rishi Saket, Jay Nandy, Balaraman Ravindran
- **학회:** KDD 2022 (ACM SIGKDD)
- **arXiv:** 2206.12626
- **코드:** https://github.com/google/vsf-time-series

## 논문 구조
1. Introduction
2. Related Work
3. Problem Formulation
4. Performance Gap Analysis (기존 모델의 VSF 성능 하락 분석)
5. Proposed Solution (kNN retrieval wrapper)
6. Experiments
7. Appendix

## 핵심 내용

### VSF 문제 정의
- 학습: 모든 N개 변수 사용
- 추론: 임의의 부분 집합 S ⊂ {1,...,N}, |S| ≪ N
- 원인: 장기 센서 고장, High→Low resource domain shift
- 기존 imputation과의 차이: 전체 시계열이 통째로 결측 (temporal locality 없음)

### 평가 프로토콜 (COMET이 따르는 표준)
- **10-seed 학습**: 서로 다른 random seed로 10번 독립 학습
- **100-mask 평가**: 각 모델에 100개 random subset S 적용
- **총 1000 runs** (10×100) → Mean(Std) 보고
- **관측 비율**: k = 15% (결측률 85%)
- **예측 길이**: Q = 12
- **데이터 분할**: train 70% / val 10% / test 20%

### 데이터셋 (4종)
| 데이터셋 | 변수 수 (N) | 설명 |
|----------|------------|------|
| METR-LA | 207 | LA 고속도로 교통 속도 |
| Solar | 137 | 알라바마 태양광 발전량 |
| Traffic | 862 | 샌프란시스코 도로 점유율 |
| ECG5000 | 140 | 심전도 |

### 평가 지표
- **MAE**: Mean Absolute Error (관측 변수 S에 대해서만 계산)
- **RMSE**: Root Mean Squared Error
- **Oracle Gap (Δ_partial)**: (E_partial - E_oracle) / E_oracle × 100

### Baseline 구성 (2행)
- **Partial**: S 변수만으로 추론 (나머지 제거) → VSF 하한
- **Oracle**: 모든 N 변수 사용, S에 대해서만 오차 계산 → 상한

### 주요 결과 (Table 1 — COMET 비교용)

#### MTGNN (Solar, pred_len=12)
| Setting | MAE | RMSE |
|---------|-----|------|
| Partial | 4.26(0.53) | 6.04(0.81) |
| Oracle | 2.94(0.27) | 4.66(0.57) |
| Δ_partial | 44.89% | 29.61% |

#### MTGNN (METR-LA, pred_len=12)
| Setting | MAE | RMSE |
|---------|-----|------|
| Partial | 4.54(0.37) | 8.90(0.68) |
| Oracle | 3.49(0.25) | 7.21(0.50) |
| Δ_partial | 30.08% | 23.43% |

#### MTGNN (ECG5000, pred_len=12)
| Setting | MAE | RMSE |
|---------|-----|------|
| Partial | 3.88(0.61) | 6.54(1.10) |
| Oracle | 3.43(0.54) | 5.94(1.08) |
| Δ_partial | 13.11% | 10.10% |

#### MTGNN (Traffic, pred_len=12)
| Setting | MAE | RMSE |
|---------|-----|------|
| Partial | 18.57(2.31) | 38.46(3.94) |
| Oracle | 11.45(0.57) | 27.48(2.14) |
| Δ_partial | 62.18% | 39.95% |

### VIDA 제안 방법
- **비파라미터 wrapper 방식**: 기존 예측 모델 위에 적용 가능
- **kNN 검색**: 관측된 S 변수의 과거 데이터로 학습 데이터에서 nearest neighbor 검색
- **결측 변수 채움**: 검색된 이웃의 N-S 변수 데이터를 빌려와 forward pass
- **Ensemble weighting**: 편향된 검색 보정을 위한 가중 앙상블
- 결과: 15% 관측 시 Oracle 성능의 ~95% 회복

### VIDA와 COMET의 핵심 차이

| 측면 | VIDA | COMET |
|------|------|-------|
| 접근 | 비파라미터 wrapper (kNN) | End-to-end 학습 모델 |
| 결측 처리 | kNN으로 학습 데이터에서 검색 후 채움 | Codebook + Decoder로 잠재 공간에서 복원 |
| 예측 모델 | 기존 모델 그대로 사용 | 복원-예측 통합 |
| 재학습 | 불필요 | 필요 (단, 단일 모델) |
| Pipeline | 2-stage (검색→예측) | End-to-end |
| 장점 | 간단, 모델 무관 | Joint optimization, 정보 병목 없음 |

### COMET 논문에서 활용할 포인트
1. **Table 1의 Partial/Oracle 값**: COMET Table 1의 Row 1, 2로 직접 인용 가능 (동일 데이터셋, 동일 프로토콜)
2. **VIDA 결과**: SOTA baseline으로 비교
3. **평가 프로토콜**: VIDA가 제안한 10-seed × 100-mask 프로토콜을 동일하게 따름
4. **데이터셋**: VIDA의 4개 데이터셋 중 Solar, METR-LA, ECG5000 사용 + ETTh1, ETTm1 추가
5. **Oracle Gap 지표**: COMET이 Oracle Gap을 얼마나 줄이는지 보고 가능

### 하이퍼파라미터 (Appendix A.3)
- seq_len (P) = 12
- pred_len (Q) = 12
- Data split: 70/10/20
- MTGNN: 학습 가능한 인접행렬, 시간 합성곱
- Missing rate: 85% (k=15%)
