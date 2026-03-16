# 3. Proposed Methodology: COMET

> 목표: 3.5페이지 (Fig 2 포함). 4개 섹션.

---

## 3.1 Overall Architecture

[Fig 2: 전체 아키텍처 다이어그램 삽입]

COMET의 파이프라인은 입력 시계열 $\mathbf{X} \in \mathbb{R}^{N \times P}$와 관측 마스크 $\mathbf{m}$으로부터 예측 $\hat{\mathbf{Y}} \in \mathbb{R}^{N \times Q}$를 생성한다:

$$\mathbf{X}, \mathbf{m} \xrightarrow{\text{Encoding}} \mathbf{Q}_{sub}, \mathbf{E}_{obs} \xrightarrow{\text{Codebook}} \mathbf{C}_{gated} \xrightarrow{\text{Decoder}} \mathbf{E}_{restored} \xrightarrow{\text{Head}} \hat{\mathbf{Y}}$$

입력 시계열은 패치 분할과 선형 투영을 거쳐 $D$차원 임베딩 공간으로 변환된다. 이후 결측 변수의 복원, codebook 조회, 최종 예측이 모두 이 임베딩 공간에서 수행된다. 원시 시계열 공간으로 되돌아가는 과정이 없으므로, 2-stage 방식의 정보 병목이 구조적으로 제거된다.

---

## 3.2 Codebook with Gating Mechanism

Codebook은 COMET의 핵심 컴포넌트로, 학습 데이터에서 관측한 정상 상태의 대표 패턴을 저장하고, 추론 시 현재 관측 상태에 맞는 패턴을 선택적으로 활용하여 결측 변수의 복원을 돕는다. 이를 위해서는 먼저 현재 관측 상태를 요약하는 벡터가 필요하다.

### Encoding과 시스템 상태 요약

입력 시계열 $\mathbf{X} \in \mathbb{R}^{N \times P}$를 길이 $p$, stride $s$의 패치로 분할하고 $D$차원에 선형 투영하여 패치 임베딩 $\mathbf{H} \in \mathbb{R}^{N \times L \times D}$를 얻는다. 여기서 $L = \lfloor(P - p)/s\rfloor + 1$은 패치 수이다. 변수별 독립적(channel-independent) multi-kernel Conv1D로 다중 스케일의 시간 패턴을 포착한 후, 관측 마스크 $\mathbf{m}$에 따라 관측 변수의 패치만 선택하여 Transformer Encoder에 입력한다. 학습 가능한 CLS 토큰을 시퀀스에 prepend하며, 그 출력 $\mathbf{Q}_{sub} \in \mathbb{R}^D$가 현재 관측 상태의 시스템 수준 요약이 된다. $\mathbf{Q}_{sub}$는 어떤 변수가 관측되었고 그 패턴이 어떠한지를 하나의 벡터로 압축한 것으로, 이후 codebook 조회의 query로 사용된다.

### 패턴 압축: Teacher Path와 EMA

$K$개의 코드북 엔트리 $\mathbf{C} \in \mathbb{R}^{K \times D}$는 시스템의 정상 상태를 대표하는 패턴 벡터이다. 학습 시 모든 변수가 관측된 데이터를 teacher path에 입력하여 전체 시스템 상태 요약 $\mathbf{Q}_{full}$을 얻고, 이를 EMA(Exponential Moving Average)로 코드북에 누적한다:

$$\mathbf{C} \leftarrow \alpha \mathbf{C} + (1 - \alpha) \tilde{\mathbf{C}}$$

$\tilde{\mathbf{C}}$는 현재 배치의 $\mathbf{Q}_{full}$로부터 가중 평균으로 계산된 업데이트이다. 이를 통해 코드북은 실제 데이터 분포의 대표 패턴을 점진적으로 학습한다. 사용 빈도가 낮은 엔트리는 활성 샘플로 교체하여 활용도를 유지한다.

### Soft Lookup과 Gating

$\mathbf{Q}_{sub}$로 코드북을 조회하여 각 엔트리 $\mathbf{c}_k$와의 관련도를 계산한다:

$$\mathbf{w}_{sub} = \text{softmax}\left(-\|\mathbf{Q}_{sub} - \mathbf{C}\|^2 / \tau\right) \in \mathbb{R}^K$$

여기서 $\tau$는 분포의 sharpness를 조절하는 온도 파라미터이다. 이 가중치를 사용하여 각 코드북 엔트리에 관련도에 비례하는 스케일을 적용한다:

$$\tilde{\mathbf{c}}_k = w_k \cdot \mathbf{c}_k, \quad k = 1, \ldots, K$$

이를 통해 현재 관측 상태와 관련 없는 패턴은 억제되고 관련 있는 패턴은 강조된 $\mathbf{C}_{gated} \in \mathbb{R}^{K \times D}$가 decoder의 cross-attention key/value로 전달된다. 이 설계는 encoder의 관측 상태 요약 품질($\mathbf{Q}_{sub}$)이 패턴 선택($\mathbf{w}_{sub}$), 복원(decoder), 예측(head)으로 직결되는 인과 경로를 형성한다. 후술할 alignment loss(Section 3.4)가 $\mathbf{Q}_{sub}$의 품질을 개선하면, 이 효과가 최종 예측 성능 향상으로 이어진다.

---

## 3.3 Two-Stage Restoration Decoder

Decoder는 결측 변수의 패치 표현을 두 단계에 걸쳐 복원한다. 결측 변수 $i$ ($m_i = 0$)에 대해, 학습 가능한 마스크 임베딩에 변수 ID 임베딩과 패치 위치 임베딩을 더하여 초기 토큰을 구성한다:

$$\mathbf{h}^{(0)}_{i,l} = \mathbf{e}_{mask} + \mathbf{e}_{var}(i) + \mathbf{e}_{pos}(l)$$

이 초기 토큰은 변수의 정체성과 시간 위치 정보를 담고 있으나, 실제 관측값에 대한 정보는 없다.

**Stage A**에서는 이 결측 변수 토큰이 관측 변수의 encoder 출력 $\mathbf{E}_{obs}$에 cross-attend하여, 관측 변수의 실시간 패턴으로부터 1차 복원을 수행한다:

$$\mathbf{h}^{(A)} = \text{CrossAttn}(\text{query}=\mathbf{h}^{(0)}_{miss},\ \text{kv}=\mathbf{E}_{obs}) + \mathbf{h}^{(0)}_{miss}$$

**Stage B**에서는 Stage A의 출력이 gating된 코드북 $\mathbf{C}_{gated}$에 cross-attend하여, 학습 과정에서 축적된 정상 패턴으로 2차 보강한다:

$$\mathbf{h}^{(B)} = \text{CrossAttn}(\text{query}=\mathbf{h}^{(A)},\ \text{kv}=\mathbf{C}_{gated}) + \mathbf{h}^{(A)}$$

Stage A가 관측 데이터만으로 수행하는 복원을, Stage B가 코드북의 사전 지식으로 보완하는 구조이다. 관측 변수에 대해서는 codebook refinement cross-attention과 원본 패치 임베딩의 residual connection을 적용한다. 복원된 전체 임베딩 $\mathbf{E}_{restored} \in \mathbb{R}^{N \times L \times D}$는 MTGNN [Wu et al., 2020] 기반의 forecast head에 입력되어 최종 예측 $\hat{\mathbf{Y}}$를 생성한다.

---

## 3.4 Training: 3-Stage Curriculum

안정적인 학습을 위해 3단계 커리큘럼 전략을 사용한다.

**Stage 1 (Warm-up)**에서는 마스킹 없이 ($r_{mask} = 0$) 전체 변수로 학습하며, task loss만 사용한다. 이 단계에서 teacher path의 $\mathbf{Q}_{full}$을 수집하여 코드북을 K-Means로 초기화한다.

**Stage 2 (Progressive Masking)**에서는 마스킹 비율과 alignment loss 가중치를 점진적으로 증가시킨다. 전체 손실 함수는 다음과 같다:

$$\mathcal{L} = \mathcal{L}_{task} + \lambda_{align} \cdot \mathcal{L}_{InfoNCE} + \lambda_{match} \cdot \mathcal{L}_{KL} + \gamma \cdot \mathcal{L}_{ent}$$

$\mathcal{L}_{InfoNCE}$는 $\mathbf{Q}_{sub}$와 동일 입력의 $\mathbf{Q}_{full}$ 간 대조 학습으로, 부분 관측에서도 전체 시스템 상태를 잘 요약하도록 encoder를 학습시킨다. $\mathcal{L}_{KL} = \text{KL}(\mathbf{w}_{sub} \| \mathbf{w}_{full})$은 코드북 조회 분포를 일치시켜, 부분 관측에서도 전체 관측과 유사한 패턴이 선택되도록 유도한다. $\mathcal{L}_{ent}$는 코드북 활용 분포의 엔트로피를 최대화하여 특정 엔트리로의 편중을 방지한다. Teacher path의 출력에는 stop-gradient를 적용하며, $\mathbf{Q}_{sub}$와 $\mathbf{Q}_{full}$의 cosine similarity가 임계치를 연속 달성하면 Stage 3으로 전이한다.

**Stage 3 (Fine-tuning)**에서는 마스킹 비율을 최대($r_{mask} = 0.85$)로 고정하고, alignment loss를 cosine annealing으로 감쇠시키며 task loss에 집중한다. Task loss는 비정규화 원본 공간에서의 MAE이며, null value ($y = 0$)를 제외한다:

$$\mathcal{L}_{task} = \frac{1}{|\mathcal{V}|} \sum_{(i,t) \in \mathcal{V}} |y_{i,t} - \hat{y}_{i,t}|$$

---

## 작성 노트
- 3.2에 인코딩 과정을 "Encoding과 시스템 상태 요약" 서브섹션으로 통합 — Q_sub 생성을 codebook 도입 직전에 설명
- $\odot$ 제거, $w_k \cdot \mathbf{c}_k$로 명확하게 표기
- 인과 체인을 별도 서브섹션에서 Soft Lookup 문단 말미로 통합
- 3.4 loss bullet list → flowing prose로 변환
- $L$의 정의를 첫 등장 시 명시: $L = \lfloor(P-p)/s\rfloor + 1$
- 3.3 말미에 forecast head 언급 추가 (파이프라인 완결)
- TODO: Fig 2 설계, citation placeholder 추가
