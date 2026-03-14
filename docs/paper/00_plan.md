# COMET 논문 구성 계획서 (Final)

## 기본 정보
- **학회:** CIKM 2026 (ACM SIGCONF)
- **분량:** 본문+부록 9페이지, References/GenAI Disclosure 별도
- **아키텍처:** 1안 (Codebook Gating)

## 기여점 (Contributions)
1. **E2E embedding-level restoration**: 시계열 복원이 아닌 임베딩 공간에서 직접 복원·예측 통합. 정보 병목/누적 오차 제거
2. **Codebook + Gating mechanism**: 정상 패턴을 EMA로 압축 저장 + 관측 상태(Q_sub) 기반 선택적 활용
3. **3-Stage Curriculum Training**: 안정적 codebook 초기화 + progressive alignment

---

## 페이지 구성 (9p)

### 1. Introduction (1.3p)
- P1: MTS 예측 중요성 + VSF 문제 동기 (센서 고장, 도메인 시프트)
- P2: 기존 한계
  - 2-stage pipeline: cascading error, objective mismatch
  - 시계열 공간 복원의 정보 병목
  - local knowledge만 전이 (temporal neighbors, pairwise correlations)
- P3: COMET 제안
  - 임베딩 공간에서 복원·예측 통합 (e2e)
  - Codebook으로 정상 패턴 압축, gating으로 선택적 활용
- P4: 기여점 3가지 bullet + 실험 하이라이트 수치
- **Fig 1 (0.3p):** VSF 문제 설정 다이어그램 (train phase vs inference phase)
  - 별도로 Fig 1-C (2-stage vs E2E 비교 다이어그램)도 준비

### 2. Problem Formulation (0.4p)
- 다변량 시계열 X ∈ R^{N×T}
- 관측 부분집합 S ⊂ {1,...,N}, 마스크 m
- 입출력 수식화, ObsMAE 정의
- 평가 프로토콜 (10-seed × 100-mask) 간략 언급

### 3. Proposed Methodology (3.5p)

#### 3.1 Overall Architecture + Fig 2 (0.8p)
- **Fig 2 (0.4p):** 전체 파이프라인 다이어그램 (codebook gating 강조)
- 핵심 메시지: "임베딩 공간에서 복원과 예측을 단일 모델로 통합"
- 각 컴포넌트 1~2문장 개요:
  - Patch Embedding: "시계열을 길이 P의 패치로 분할, D차원 투영"
  - CI-Conv1D: "multi-kernel(1,3,5) 합성곱으로 변수별 독립 시간 패턴 추출"
  - Encoder: "관측 변수 패치에 TransformerEncoder 적용, CLS 토큰으로 시스템 상태 요약 Q_sub 생성"
  - MTGNN Head: "학습 가능 그래프 + dilated inception [Wu et al., 2020]으로 최종 예측"

#### 3.2 Codebook with Gating Mechanism (1.2p) ← 핵심
- Codebook 정의: C ∈ R^{K×D}, 정상 패턴 저장소
- 패턴 압축 (Teacher Path): Q_full → EMA → C 업데이트
- Soft lookup: Q_sub × C → w_sub (거리 기반, softmax, 온도 τ)
- **Gating: C_gated = w_sub ⊙ C** ← 핵심 수식
- 인과 체인: Q_sub → w_sub → C_gated → decoder → 예측
- Dead entry revival, entropy regularization 간략 언급

#### 3.3 Two-Stage Restoration Decoder (0.7p)
- 결측 변수 토큰 초기화 (mask embedding + var ID + pos)
- Stage A: cross-attn(miss tokens, obs patches) — 데이터 기반 복원
- Stage B: cross-attn(stage A output, C_gated) — 지식 기반 보강
- 관측 변수: codebook refinement + residual
- **임베딩 공간 복원 강조** (시계열로 돌아가지 않음 → 정보 병목 없음)

#### 3.4 Training Strategy (0.8p)
- 3-Stage Curriculum 개요:
  - Stage 1 (warm-up): mask=0, task loss only, Q_full 수집 → K-Means init
  - Stage 2 (progressive): mask↑, alignment loss↑ (InfoNCE + KL Match)
  - Stage 3 (fine-tuning): mask=max, alignment cosine decay
- Loss 수식: L = L_task + λ_align·InfoNCE + λ_match·KL + γ·EntropyReg
- Teacher stop-gradient, denormalized loss, null value masking 간략 언급

### 4. Experiments (2.5p)

#### 4.1 Experimental Setup (0.4p)
- **Datasets (5종):**

| Dataset | N | Timesteps | Sample Rate |
|---------|---|-----------|-------------|
| Solar | 137 | 52,560 | 10min |
| METR-LA | 207 | 34,272 | 5min |
| ECG5000 | 140 | 5,000 | - |
| ETTh1 | 7 | 17,420 | 1h |
| ETTm1 | 7 | 69,680 | 15min |

- **Metrics:** ObsMAE, ObsRMSE (denormalized, null_val=0.0 제외)
- **Protocol:** 10-seed × 100-mask, 결측률 85%, pred_len=12
- **Baselines:** VIDA, FDW, CSDI, SAITS, SSGAN, KNNE, IIM, MICE, TRMF
- **Implementation:** d_model=128, K=16, lr=1e-3, batch=64

#### 4.2 RQ1: Main Comparison (0.7p) — Table 1
- **구성 (행):**
  - MTGNN Partial (하한, Chauhan et al. 인용)
  - MTGNN Oracle (상한, Chauhan et al. 인용)
  - COMET w/o Codebook (e2e 아키텍처 기여 분리)
  - **COMET (Ours)** (전체 파이프라인)
  - 기존 SOTA: VIDA, FDW, CSDI, SAITS 등
- **구성 (열):** 5 datasets × MAE/RMSE
- 스토리: Partial→COMET nocb = e2e 기여, nocb→COMET = codebook 기여

#### 4.3 RQ2: Baseline Comparison (0.5p) — Table 1에 통합 또는 별도 표
- imputation baselines (CSDI, SAITS, SSGAN, MICE 등) 대비 성능
- VIDA 대비 성능 비교

#### 4.4 RQ3: Ablation Study (0.4p) — Table 2
- w/o Gating (원본 아키텍처 — codebook 있지만 gating 없음)
- w/o Embedding (ts_input — 시계열 직접 입력)
- K sensitivity (4, 8, 16, 32) — 간결하게 1~2줄
- → gating 기여 + e2e embedding 기여 정량 분리

#### 4.5 RQ4: Missing Rate Sensitivity (0.2p) — Fig 3
- 50%, 65%, 75%, 85%, 95% 결측률별 ObsMAE 곡선
- COMET vs nocb (vs baseline 가능하면)

#### 4.6 RQ5: Codebook Analysis (0.3p) — Fig 4
- t-SNE: codebook entry 시각화 (패턴 클러스터링)
- w_sub gating weight 분포 (어떤 패턴이 선택되는지)

### 5. Related Work (0.8p)
- MTS Forecasting (PatchTST, iTransformer, MTGNN 등)
- Variable Subset Forecasting (Chauhan et al., VIDA, GIMCC, GinAR, TOI-VSF)
- Codebook / VQ in Time Series (VQ-VAE, TimeVQVAE)

### 6. Conclusion (0.3p)
- 기여 재요약, 한계, 향후 연구

### GenAI Disclosure (페이지 미포함)
### References (페이지 미포함)

---

## Figure/Table 목록

| # | 유형 | 내용 | 위치 | 예상 크기 |
|---|------|------|------|----------|
| Fig 1 | 다이어그램 | VSF 문제 설정 (train vs inference) | Intro | 0.3p |
| Fig 1-C | 다이어그램 (예비) | 2-stage vs E2E 비교 | Intro 대체용 | 0.3p |
| Fig 2 | 아키텍처 | COMET 전체 파이프라인 (gating 강조) | Method 3.1 | 0.4p |
| Fig 3 | Line plot | Missing rate sensitivity 곡선 | Exp 4.5 | 0.2p |
| Fig 4 | Scatter/Heatmap | Codebook t-SNE + w_sub 분포 | Exp 4.6 | 0.3p |
| Tab 1 | 표 | RQ1+RQ2: Main + baseline comparison | Exp 4.2-4.3 | 0.4p |
| Tab 2 | 표 | RQ3: Ablation (gating, embedding, K) | Exp 4.4 | 0.2p |

---

## 참고 논문 (핵심)

### VSF
- Chauhan et al., "Multi-Variate Time Series Forecasting on Variable Subsets", KDD 2022
- Liang et al., "VIDA: Imputation via Domain Adaptation", KDD 2025
- GIMCC: "Generative Imputation with Multi-level Causal Consistency", KDD 2025
- GinAR (2023)
- TOI-VSF: "Is Precise Recovery Necessary?", arXiv 2024

### MTS Forecasting
- MTGNN (Wu et al., 2020)
- PatchTST (Nie et al., 2023)
- iTransformer (Liu et al., 2024)

### Codebook
- VQ-VAE (van den Oord et al., 2017)
- TimeVQVAE (Lee et al., 2023)

### Imputation Baselines
- CSDI (Tashiro et al., 2021), SAITS (Du et al., 2023), BRITS (Cao et al., 2018)

---

## 작업 순서

| # | 작업 | 파일 | 상태 |
|---|------|------|------|
| 1 | 구성 계획서 | `00_plan.md` | ✅ |
| 2 | 각 섹션 구체 내용 논의 + 작성 | `01~06_*.md` | ⏳ |
| 3 | Figure 설계안 | `figures/` | ⏳ |
| 4 | 참고문헌 BibTeX | `references.bib` | ⏳ |
