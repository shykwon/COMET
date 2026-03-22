# Codebook Ablation 실험 결과

## 개요
- 데이터셋: Solar (N=137, seq_len=12, pred_len=12)
- 프로토콜: seed 0, 100-mask eval
- Head: MTGNN
- 기본 설정: K=16, bs=64, amp_bf16

## Ablation v1 (disable_ema=true — 버그)

> 주의: default.yaml에 disable_ema=true가 초기 커밋부터 있어서, 모든 실험이 EMA OFF 상태로 진행됨.
> Baseline이 사실상 No EMA 조건. C, E와의 비교가 무의미.

| # | 실험 | ObsMAE | ObsRMSE | EMA | 비고 |
|---|------|--------|---------|:---:|------|
| - | Baseline (EMA OFF) | 2.1023 | 3.4749 | OFF | disable_ema=true 버그 |
| A | Hard Lookup (EMA OFF) | 2.0787 | 3.4698 | OFF | |
| B | No Revival (EMA OFF) | 2.1469 | 3.5233 | OFF | EMA OFF면 revival도 자동 비활성 → Baseline과 동일 조건 |
| C | No EMA (EMA OFF) | 2.1028 | 3.4667 | OFF | Baseline과 동일 조건 |
| E | No Entropy (EMA OFF) | 2.0870 | 3.4814 | OFF | ppl=1.0 (codebook 붕괴) |

**v1 결론:** EMA OFF 상태에서도 K-Means init만으로 성능이 유지됨 (ObsMAE ~2.08-2.10).

---

## Ablation v2 (disable_ema=false — 정상)

### 실험 조건

| # | 실험 | lookup | Stage B | EMA | entropy | 태그 |
|---|------|--------|---------|:---:|:-------:|------|
| 1 | Baseline | soft | cross-attention | ON | ON | (없음) |
| 2 | Hard+EMA (VQ-VAE) | hard (ST) | cross-attention | ON | ON | `_hardlk` |
| 3 | No Entropy+EMA | soft | cross-attention | ON | OFF | `_noent` |
| 4 | Soft+Direct Add | soft | direct add | ON | ON | `_dadd` |
| 5 | Hard+Direct Add | hard (ST) | direct add | ON | ON | `_hardlk_dadd` |

### 결과 (Solar seed 0, 100-mask)

| # | 실험 | ObsMAE | ObsRMSE | vs Baseline | ppl |
|---|------|--------|---------|-------------|-----|
| 1 | **Baseline** (soft+cross-attn+EMA) | **2.0845** | 3.4749 | - | ~14 |
| 2 | Hard+EMA (VQ-VAE) | 2.0895 | 3.4698 | +0.2% | ~14 |
| 3 | No Entropy+EMA | 2.0881 | 3.4814 | +0.2% | ~1 (붕괴) |
| 4 | Soft+Direct Add | 2.6972 | - | **+29.4%** | - |
| 5 | Hard+Direct Add | 2.6418 | - | **+26.7%** | - |

### 핵심 발견

1. **Soft vs Hard 차이 없음** (1 vs 2: 2.0845 vs 2.0895, +0.2%)
   - Hard lookup(VQ-VAE 방식)으로 전환해도 성능 손실 없음
   - 모델 간소화 가능

2. **Entropy Reg 불필요** (1 vs 3: 2.0845 vs 2.0881, +0.2%)
   - entropy 없으면 ppl=1 (붕괴)이지만 성능은 동일
   - EMA + revival이 codebook 활용도를 유지시키는 것으로 추정

3. **Cross-Attention이 핵심** (1 vs 4: 2.0845 vs 2.6972, +29.4%)
   - Direct add로 대체하면 성능 급락
   - query-dependent 전달이 codebook 활용의 핵심
   - cross-attention의 W_v projection이 아니라, query에 따라 다른 활용이 중요

4. **Direct Add에서도 Hard > Soft** (4 vs 5: 2.6972 vs 2.6418)
   - Direct add에서는 hard가 소폭 우세
   - 하지만 둘 다 cross-attention 대비 크게 열등

---

## 추가 실험 (진행 중/예정)

### FiLM (진행 중)
- Soft + FiLM: cross-attention과 direct add 사이 복잡도
- Hard + FiLM: hard selection + FiLM modulation
- 태그: `_film`, `_hardlk_film`

### Hard NoGrad (진행 중)
- Hard lookup에서 ST(straight-through) 제거
- Selection을 통한 gradient 차단
- Encoder는 InfoNCE/KL로만 학습, codebook은 EMA로만 학습
- 태그: `_hardng`
- 기대: Hard ST(2.0895)와 비슷하면 → ST 불필요, 모델 더 간소화

---

## 간소화 방향 (ablation 기반)

```
현재:
  soft lookup → gating(w*C) → MHA cross-attention + FFN + entropy reg
  하이퍼파라미터: τ, entropy_reg_weight

간소화:
  hard lookup (no grad) → C[k*] 선택 → MHA cross-attention + FFN
  제거: soft lookup, gating 연산, entropy reg, τ
  유지: cross-attention (query-dependent 전달), EMA, K-Means init
```

---

## 복잡도 순서 (적용 방식별)

| 적용 방식 | 추가 파라미터 | ObsMAE | 결론 |
|----------|-------------|--------|------|
| Direct Add | 0 | ~2.67 | 불충분 |
| FiLM | ~33K | TBD | |
| Cross-Attention | ~66K | ~2.08 | 필수 |

---

## 파일 경로

- v1 결과: `logs/comet_solar_K16_conv1d_{tag}_s0_20260320_*`
- v2 결과: `logs/comet_solar_K16_conv1d_{tag}_s0_20260321_*`
- 스크립트: `scripts/run_codebook_ablation.sh`
- 로그: `logs/codebook_ablation_v2.log`
