# 1. Introduction

> 목표: 1.3페이지 (Fig 1 포함). VSF 동기 → 기존 한계 → COMET 제안 → 기여 요약

---

## P1: MTS 예측의 중요성 + VSF 문제 제기

다변량 시계열 예측(MTSF)은 교통[Li et al., 2018], 에너지[Solar], 의료[ECG] 등 지능형 시스템의 핵심 과제이다. 최신 MTSF 모델들은 변수 간 의존성을 활용하여 높은 예측 정확도를 달성하지만, 학습과 추론 시점에 모든 변수가 가용하다는 전제에 의존한다. 그러나 실제 배포 환경에서는 센서 고장, 통신 장애, 비용 제약 등으로 인해 추론 시 임의의 변수 부분집합만 관측 가능한 상황이 빈번히 발생한다. Chauhan et al. [2022]은 이러한 문제를 Variable Subset Forecasting (VSF)으로 정의하고, 기존 모델의 성능이 40% 이상 하락함을 보였다.

[→ Fig 1: VSF 문제 설정 다이어그램]

## P2-a: 기존 보간(Imputation) 방법의 한계

결측 변수를 채워넣는 전통적인 방법은 시계열 보간(imputation) 기법 [CSDI, SAITS, BRITS 등]을 적용하는 것이다. 이들은 같은 변수의 인접 시점이나 다른 변수의 동시점 값을 참조하여 결측값을 추정하는데, 이는 특정 시점의 값이 산발적으로 누락된 경우에만 유효하다. VSF에서는 변수의 전체 시계열이 통째로 부재하므로, 참조할 시간적 이웃 자체가 존재하지 않는다. 또한 학습 시 전체 변수로 구축된 변수 간 상관 구조가, 추론 시 임의의 부분집합만 관측되면 근본적으로 무너진다.

## P2-b: VSF 전용 연구의 한계

이러한 한계를 인식하고, 검색 기반 [Chauhan et al., 2022], 보간 기반 [GIMCC, 2025; TOI-VSF, 2024], 도메인 적응 [VIDA, 2025] 등 VSF에 특화된 다양한 접근이 제안되었다. 이들은 의미 있는 성능 개선을 달성했으나, 대부분 결측 변수를 원시 시계열 공간에서 복원한 뒤 별도의 예측 모델에 입력하는 2-stage 구조를 따른다. GinAR [Yu et al., 2024]은 end-to-end 구조를 시도하였으나, 학습 데이터에서 축적한 시스템 수준의 정상 패턴을 명시적으로 저장·활용하는 메커니즘이 없다. 결과적으로 기존 연구들은 **(i)** 복원 오차의 예측 단계 전파, **(ii)** 복원 목적과 예측 목적 간 불일치, **(iii)** 원시 공간 복원 시 잠재 표현의 손실 중 하나 이상의 한계를 수반한다.

## P3: COMET 제안

본 논문은 이러한 2-stage 구조의 한계를 근본적으로 해소하는 COMET (**CO**debook-augmented **M**ultivariate time-series forecasting with **E**xpertise **T**ransfer)을 제안한다. COMET은 변수 복원과 예측을 하나의 모델에서 수행하는 end-to-end 프레임워크로, 전 과정이 임베딩 공간에서 이루어진다. 관측 변수를 패치 임베딩으로 인코딩하고, 결측 변수를 임베딩 수준에서 복원하며, 복원된 임베딩으로부터 직접 예측을 생성한다. 원시 시계열 공간으로 되돌아가는 과정이 없으므로 앞서 지적한 세 가지 한계가 구조적으로 제거된다.

COMET의 핵심은 **codebook gating 메커니즘**이다. 학습 시 전체 변수가 관측된 데이터로부터 시스템의 정상 상태 패턴을 codebook에 압축 저장한다. 추론 시에는 encoder가 현재 관측 상태를 요약하고, 이 요약을 기반으로 codebook에서 관련 패턴을 선택적으로 활성화(gating)하여 decoder에 전달한다. 이를 통해 관측 상태의 요약 품질이 패턴 선택, 복원, 예측 성능으로 직결되는 인과 경로가 형성된다.

## P4: 기여 요약

본 논문의 기여는 다음과 같다:

- **End-to-end 임베딩 수준 복원.** 변수 복원과 예측을 임베딩 공간에서 통합하는 아키텍처를 제안한다. 원시 시계열 공간을 거치지 않으므로 2-stage 파이프라인의 누적 오차, 목적 함수 불일치, 정보 병목을 구조적으로 제거한다. (Section 3.1, 3.3)

- **Codebook gating 메커니즘.** 학습 데이터의 정상 패턴을 codebook에 압축하고, 추론 시 관측 상태에 따라 관련 패턴만 선택적으로 활용하는 메커니즘을 도입한다. 관측 요약에서 예측까지 직접적인 인과 경로를 확립한다. (Section 3.2)

- **3단계 커리큘럼 학습.** codebook의 안정적 초기화와 부분-전체 관측 표현 간 점진적 정렬을 보장하는 학습 전략을 설계한다. (Section 3.4)

5개 실측 데이터셋에 대한 실험에서 COMET은 최신 VSF 방법인 VIDA [Liang et al., 2025] 대비 최대 X%의 성능 개선을 달성하며, 추론 시 단일 forward pass로 동작한다.

---

## Figure 설계안

### Fig 1 (A안: VSF 문제 설정) — 본문 사용
```
[Training Phase]                    [Inference Phase]
V1 ████████████ → predict          V1 ████████████ → predict
V2 ████████████ → predict          V2 (missing)     → predict?
V3 ████████████ → predict          V3 ████████████ → predict
V4 ████████████ → predict          V4 (missing)     → predict?
V5 ████████████ → predict          V5 (missing)     → predict?
   ← lookback →  ← horizon →         ← lookback →  ← horizon →
   All N variables observed           Only S ⊂ N observed (|S| ≪ N)
```

### Fig 1-C (2-stage vs E2E 비교) — 예비
```
[Two-Stage Pipeline]
x_obs → Impute → x̂_full (R^{N×T}) → Forecast Model → ŷ
         ↓                              ↓
       L_recon (복원 목적)            L_pred (예측 목적)
       ← 독립 최적화, 누적 오차 →

[COMET: End-to-End]
x_obs → [Embed → Encode → Codebook+Gating → Decode → Head] → ŷ
         └────── 임베딩 공간 (R^{N×L×D})에서 복원+예측 통합 ──┘
                    L = L_task + L_align (joint 최적화)
```

## 검토 사항 / TODO
- [ ] **P2-b: 기존 VSF 연구 구조적 정리** — FDW, VIDA, GIMCC 외에 GinAR, TOI-VSF 등 추가 조사. 연구들을 접근 방식별로 분류 (검색 기반 / 도메인 적응 / 생성 모델 등)하여 체계적으로 정리
- [ ] **P3: 모델 내용 보강 평가** — e2e 구조와 codebook gating 외에, decoder의 two-stage 복원(관측 패치 → codebook), encoder의 역할 등 방법론적 내용을 어느 수준까지 언급할지 결정. Introduction에서 너무 상세하면 Methodology와 겹치고, 너무 간략하면 기여가 모호해짐
- [ ] 마지막 문장 "X%" → 실험 결과 확정 후 채움
- [ ] "To the best of our knowledge" — e2e embedding-level VSF가 최초라면 사용 가능
