# VIDA 논문 분석 요약

## 서지정보
- **제목:** Imputation via Domain Adaptation: Rethinking Variable Subset Forecasting from Knowledge Transfer
- **저자:** Runchang Liang, Qi Hao, Yue Gao, Kunpeng Liu, Lu Jiang, Pengyang Wang, Minghao Yin
- **학회:** KDD 2025 (ACM SIGKDD)
- **DOI:** 10.1145/3711896.3737007
- **코드:** https://github.com/liangrc800/vida-vsf

## 논문 구조
```
1. Introduction
2. Problem Formulation
3. Methodology (3.1 Overview, 3.2 Stage I: Source Pretraining, 3.3 Stage II: Feature Alignment, 3.4 Training & Inference)
4. Experiments (RQ1~RQ5)
5. Related Work
6. Conclusion
```
→ **Related Work가 Experiments 뒤에 위치** (COMET도 이 구조 채택)

## 핵심 내용

### VSF 재정의: Domain Adaptation 문제
- Source domain D_N: 전체 N 변수 (학습)
- Target domain D_S: 부분 S 변수 (추론), D_S ⊂ D_N, |S| ≪ |N|
- 기존 imputation의 한계: local knowledge만 전이 (temporal neighbors, pairwise correlations)
- VIDA: **global time-frequency dynamics** + **domain-invariant representation** 전이

### 3 Pillars
1. **Global Time-Frequency Knowledge Preservation**: TCN (dilated conv) + FFT + Fourier Neural Operator (저주파 spectral consistency)
2. **Sinkhorn-Regularized Distribution Alignment**: Optimal Transport 기반, source-target feature 분포 정렬
3. **Task-Driven Consistency Distillation**: 예측 일관성 + knowledge distillation (reconstruction accuracy가 아닌 forecasting 성능 직접 최적화)

### 2-Stage 학습
- **Stage 1 (Source Pretraining):** 전체 데이터로 시간-주파수 표현 학습, self-supervised prediction consistency
  - L_stage1 = α·L_spc + β·L_spfc
- **Stage 2 (Feature Alignment):** 마스킹된 subset → source domain과 feature alignment
  - L_stage2 = γ·L_align (Sinkhorn) + λ·L_fafc
  - Forecasting model은 frozen, encoder-decoder만 재학습

### 추론
1. Encoder: 관측 변수 → time-frequency representation
2. Decoder: 전체 변수 공간으로 복원
3. Frozen forecasting model: 예측

## 핵심 결과 (Table 1)

### MTGNN backbone (4 datasets, pred_len=12)

| Dataset | Setting | MAE | RMSE |
|---------|---------|-----|------|
| **SOLAR** | Partial | 4.36(0.53) | 6.04(0.81) |
| | Oracle | 2.94(0.27) | 4.66(0.57) |
| | **+VIDA** | **2.33(0.25)** | **3.71(0.50)** |
| | Δ_subset | 46.55% | 38.57% |
| | Δ_complete | 20.74% | 20.38% |
| **METR-LA** | Partial | 4.54(0.37) | 8.90(0.68) |
| | Oracle | 3.49(0.25) | 7.21(0.50) |
| | **+VIDA** | **3.36(0.20)** | **6.76(0.42)** |
| | Δ_subset | 25.99% | 24.04% |
| **ECG5000** | Partial | 3.88(0.61) | 6.54(1.10) |
| | Oracle | 3.43(0.54) | 5.94(1.08) |
| | **+VIDA** | **3.22(0.58)** | **5.47(1.11)** |
| | Δ_subset | 14.43% | 16.36% |
| **TRAFFIC** | Partial | 18.67(2.31) | 38.46(3.94) |
| | Oracle | 11.45(0.57) | 27.48(2.14) |
| | **+VIDA** | **11.36(0.71)** | **27.84(2.14)** |
| | Δ_subset | 39.15% | 27.61% |

### ASTGCN, MSTGCN, TGCN 에서도 동일 패턴
- 평균 Δ_subset: SOLAR 47.97%, METR-LA 20.03%, ECG 16.05%, Traffic 26.06%
- **Solar에서 가장 큰 개선** (Oracle 초과 성능: Δ_complete=20.74%)

### Imputation baselines 비교 (RQ2)
- KNNE, IIM, TRMF, CSDI, SAITS, SSGAN, MICE, FDW 대비 VIDA 우세
- 평균 25% 개선

## VIDA vs COMET 비교

| 측면 | VIDA | COMET |
|------|------|-------|
| VSF 관점 | Domain adaptation | End-to-end restoration + forecasting |
| 복원 방식 | Time-frequency reconstruction | Codebook + Two-stage decoder |
| 지식 전이 | Sinkhorn OT alignment | EMA codebook + gating |
| 학습 | 2-stage (pretrain → align) | 3-stage curriculum |
| Forecasting model | 기존 모델 frozen | 통합 모델 joint training |
| Pipeline | 2-stage (encoder-decoder → forecast) | End-to-end |
| 핵심 차별점 | Frequency domain 활용, OT 정렬 | Codebook 패턴 압축, gating 선택 |
| 추론 비용 | Encoder+Decoder+Forecast model | 단일 forward pass |

### COMET 논문에서 VIDA 대비 어필할 점
1. **True end-to-end**: VIDA는 encoder-decoder + frozen forecast model = 여전히 2-stage. COMET은 단일 모델
2. **Codebook의 해석 가능성**: EMA로 수집된 패턴은 t-SNE로 시각화 가능. VIDA의 latent representation은 해석 어려움
3. **추론 효율성**: COMET은 단일 forward pass, VIDA는 encoder-decoder + forecast model 2번 실행
4. **Gating의 명시적 패턴 선택**: COMET은 어떤 codebook entry가 활성화됐는지 명시적. VIDA의 alignment는 암묵적

### COMET이 VIDA 결과를 인용할 때
- Table 1의 **MTGNN +VIDA** 값을 baseline으로 비교 가능
- 동일 데이터셋 (Solar, METR-LA, ECG5000, Traffic), 동일 프로토콜 (10-seed × 100-mask)
- **VIDA가 Solar에서 Oracle을 초과 (MAE 2.33 < Oracle 2.94)** — 이것이 가능한 이유와 COMET 성능 비교 필요

## Writing Style 특징

### 논문 구조
- Introduction → Problem Formulation → Methodology → Experiments → Related Work → Conclusion
- **Related Work가 뒤에 위치** (COMET도 동일 구조 채택)

### 기여점 표기
```
In summary, our contributions can be summarized as follows:
• A Knowledge Transfer Reformulation for VSF: ...
• The VIDA Framework: ...
• Extensive Empirical Validation: ...
```
- bullet point (•) 사용, 각 기여점에 bold 제목 + 설명

### 수식 패턴
- 매우 상세: 텐서 shape, 연산, 변수 정의를 빠짐없이 명시
- 수식 직후 "where ..." 로 즉시 해석
- 총 31개 수식 (12페이지 기준) — 수식 밀도가 높음
- Stage별로 loss를 분리 정의 후 최종 합산

### 실험 구성
- RQ (Research Question) 형식으로 실험 동기 부여
  - RQ1: Overall performance
  - RQ2: Imputation baseline 비교
  - RQ3: Ablation study
  - RQ4: Alignment method 비교
  - RQ5: Hyperparameter sensitivity
- 표: Mean(Std) + Δ_subset/Δ_complete 지표

### "Pillar" 프레이밍
- 3개 핵심 요소를 P1, P2, P3로 명명하여 구조적으로 정리
- COMET도 유사하게 핵심 요소를 번호로 정리 가능

### 핵심 writing 기법
1. **VSF를 domain adaptation으로 재정의** — 기존 관점 전환 (reframing)
2. **"three key mechanisms"** 식으로 설계 근거를 번호 리스트로 정당화
3. **Δ 지표 2종** (Δ_subset: Partial 대비 개선, Δ_complete: Oracle 대비 개선)
4. **Figure 2**: 전체 아키텍처 다이어그램 (2-stage 구조 명확히 표현)
5. **하이퍼파라미터 heatmap**: α, β, γ, λ에 대한 grid search 결과 시각화
