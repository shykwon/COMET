# VIDA 논문 Writing Style 분석

## 1. 논문 구조 패턴

```
1. INTRODUCTION (~1.5 pages)
2. RELATED WORK (~0.5 pages)
3. PROBLEM FORMULATION (~0.3 pages)
4. PERFORMANCE GAP ANALYSIS (~1.5 pages) ← 독특: 기존 방법의 실패를 먼저 보여줌
5. PROPOSED SOLUTION (~1.5 pages)
6. EMPIRICAL EVALUATION (~1 page)
7. RETRIEVAL ABLATIONS (~0.5 pages)
8. WEIGHTING ABLATIONS (~0.5 pages)
9. COMPARISON TO DATA IMPUTATION (~0.5 pages)
10. CORRELATED FAILURES (~0.5 pages)
11. CONCLUSION (~0.3 pages)
```

**특이점:** Related Work가 Introduction 바로 뒤에 온다 (COMET과 다름). 실험이 여러 섹션으로 분산됨 (6,7,8,9,10).

## 2. Introduction 전개 패턴

### 흐름: 문제 동기 → 시나리오 구체화 → 기존 방법 한계 → 제안 → 기여점

**P1:** MTS 예측의 중요성 + 실무 시나리오 제시 (1문장으로 간결하게)
```
"We formulate a new inference task... called Variable Subset Forecast (VSF),
where only a small subset of the variables is available during inference."
```

**P2-P3:** 두 가지 구체적 시나리오로 동기 부여 (번호 리스트)
1. Long Term Variable Data Unavailability (센서 고장)
2. High→Low Resource Domain Shift (대형→소규모 병원)

→ **기법: 추상적 문제를 구체적 산업 사례로 풀어냄**

**P4:** 기존 방법의 한계 (imputation은 temporal locality에 의존 → VSF에 부적합)

**P5:** VSF 문제의 두 가지 핵심 챌린지:
1. N-S 변수 데이터 부재
2. 변수 간 상호작용 활용 불가

**P6:** 제안 방법 요약 + 기여점 3가지 (번호 리스트)
```
We make the following contributions in this paper:
(1) We formulate a new inference task called Variable Subset Forecast...
(2) We propose a novel wrapper solution...
(3) We conduct extensive experiments...
```

### Writing 기법
- **기여점은 반드시 번호 리스트** (1), (2), (3)로 명시
- 각 기여점 끝에 **(Section X)** 참조 추가
- "To the best of our knowledge"로 novelty 주장
- Figure 1을 Introduction에 배치 (문제 설명용 다이어그램)

## 3. Problem Formulation 수식 패턴

### 표기법
| 기호 | 의미 |
|------|------|
| $N$ | 전체 변수 수 (스칼라) |
| $\mathbf{N}$ | 변수 인덱스 집합 $[[1...N]]$ |
| $S$ | 관측 부분 집합 ($S \subset \mathbf{N}$, $|S| \ll |N|$) |
| $P$ | 입력 시점 수 |
| $Q$ | 예측 시점 수 (horizon) |
| $D$ | feature 차원 |
| $\mathbf{Z}_{t_i}$ | 시점 $t_i$의 전체 변수 행렬 $\in \mathbb{R}^{N \times D}$ |
| $f$ | 예측 모델 $f: X \rightarrow Y$ |

### 수식 스타일
- **간결함 우선**: 복잡한 수식 없이 집합 표기와 텐서 shape으로 정의
- shape을 괄호 안에 명시: "a matrix in $\mathbb{R}^{N \times D}$"
- 입출력을 시퀀스 표기: $X = \{Z_{t_1}, Z_{t_2}, ..., Z_{t_P}\}$
- "S is chosen completely at random" — 핵심 제약을 자연어로 강조

### COMET에 적용할 점
- Problem Formulation은 짧게 (0.3~0.5 page)
- 수식보다 setup 설명에 집중
- 핵심 constraint ("random subset", "no temporal locality")를 명확히 문장으로

## 4. 실험 구성 패턴

### Baseline 구성 (2행 bound)
| Row | 이름 | 설명 | 역할 |
|-----|------|------|------|
| Partial | S만으로 추론 | 하한 |
| Oracle | N 전체 사용, S로 평가 | 상한 |

→ **COMET도 동일 구조 + 중간 행 추가 (w/o codebook, full COMET)**

### 지표 표기
```
MAE = (1/|S|) Σ |Ŷ - Y|
RMSE = sqrt((1/|S|) Σ (Ŷ - Y)²)
Δ_partial = (E_partial - E_oracle) / E_oracle × 100
```
- **Δ (Oracle Gap)** 지표가 핵심 — 절대값보다 Oracle 대비 상대 성능으로 기여를 보여줌
- 괄호 안에 std 표기: `4.26(0.53)`

### 표 구성
- Table 1: 4 datasets × 5 models × 2 settings (Partial, Oracle) — **가장 중요한 표**
- Table 4: 제안 방법의 Δ_ensemble — Δ_partial 대비 개선 보여줌
- 나머지 Table: ablation 하나씩 독립 표로

### COMET에 적용할 점
- **Oracle Gap (Δ) 지표를 도입**: COMET이 Oracle gap을 얼마나 줄이는지가 핵심 메시지
- 표 형식: `Mean(Std)` with Δ 값
- 각 ablation은 독립 표 + 분석 문단

## 5. Method 설명 패턴

### 흐름: 직관 → 수식 → 해석

```
(직관) "The underlying idea is that - with the original test instance data of S
        and the borrowed data for N−S, the spatial module can make much more
        informed decision."

(수식) D(X', X'') = (1/P*|S|*D) ΣΣΣ |X'_{p,s,d} - X''_{p,s,d}|^b

(해석) "where b is the exponent factor. We also observed in our ablation
        experiments that b=0.5 works best."
```

### 수식 스타일 특징
- **매우 구체적**: 텐서 인덱싱, shape, 연산을 빠짐없이 명시
- **즉시 해석**: 수식 바로 뒤에 "where..."로 변수 설명
- **실용적 코멘트**: "we observed that b=0.5 works best" — 이론보다 실험적 선택 강조
- **알고리즘 블록** (Algorithm 1): 복잡한 절차는 pseudocode로 정리

### 수식 번호 활용
- 핵심 수식만 번호 부여 (전체 11페이지에 수식 10개 미만)
- 본문에서 "as in equation 4", "using Eq 8"로 참조
- inline 수식은 번호 없이 자연스럽게 삽입

## 6. 논리 전개 기법 (Writing Techniques)

### (1) "먼저 문제를 보여주고, 그 다음 해결책" 패턴
- Section 4 (Performance Gap Analysis)에서 기존 모델의 실패를 정량적으로 증명
- Section 5에서 "In the previous section, we showed that..."로 연결
- **COMET 적용:** Introduction에서 2-stage pipeline 한계 → Methodology에서 e2e 해결

### (2) "질문 → 실험 → 답" 패턴
```
"A natural question that arises is - what if we completely remove the spatial
 module and let the models learn solely based on temporal patterns?"
→ 실험 설계 → 결과 표 → 해석
```
- ablation을 질문으로 동기 부여
- **COMET 적용:** "What if we remove the codebook gating?" → ablation 결과

### (3) "Simple yet effective" 어필
```
"We propose a simple, yet effective, wrapper technique"
"The algorithm is very simple to code"
```
- 방법의 간결함을 장점으로 강조
- **COMET 적용:** gating은 한 줄 코드, 추가 파라미터 0개 — 이걸 강조할 것

### (4) 정량적 claim
```
"recover close to 95% of the best-case performance even with only 15%
 of the variables available"
```
- 추상적 주장 대신 구체적 수치로 기여 표현
- **COMET 적용:** "reduces Oracle gap by X% compared to baseline"

### (5) "For completeness" / "Although infeasible in practice"
- 보조 실험을 넣을 때 사용하는 표현
- Oracle baseline이 실제 불가능함을 명시하면서도 비교 대상으로 포함

## 7. Figure 활용 패턴

| Figure | 위치 | 용도 |
|--------|------|------|
| Fig 1 | Introduction | 문제 설정 다이어그램 (train vs test) |
| Fig 3 | Evaluation | Reciprocal Rank 분포 (데이터셋별 난이도 차이 설명) |
| Fig 4 | Ablation | 하이퍼파라미터 변화에 따른 성능 변화 (bar + line) |
| Fig 5 | Appendix | Horizon별 Oracle-Partial gap 변화 |
| Fig 6 | Appendix | Variable importance vs performance correlation |

- **아키텍처 다이어그램 없음** (wrapper 방식이라 모델 구조가 단순)
- **scatter plot으로 correlation 보여주기** (Fig 6) — 정성적 분석에 유용
- **COMET은 아키텍처가 복잡하므로 Fig 2에 전체 파이프라인 다이어그램 필수**

## 8. COMET 논문에 적용할 핵심 Writing 원칙

1. **기여점은 번호 리스트 (1)(2)(3)** + Section 참조
2. **Oracle Gap (Δ) 지표** 도입 — 절대값보다 상대 개선 강조
3. **"먼저 문제를 보여주고 해결"** 패턴으로 각 섹션 동기 부여
4. **수식: 직관 → 수식 → 해석** 순서 엄수
5. **간결한 수식** — shape과 인덱스를 명확히, 불필요한 복잡성 제거
6. **"simple yet effective"** 프레이밍 — gating의 간결함 강조
7. **구체적 수치로 claim** — "reduces gap by X%"
8. **각 ablation을 질문으로 시작** — "What if we remove X?"
9. **Table 구성**: 핵심 결과 표 1개 + ablation 표 여러 개
10. **std를 괄호 표기**: `Mean(Std)` 형식
