# COMET 실험 설계

## 1. 기본 실험 환경 (Experimental Setup)

- **데이터셋 (4종):** Solar(137), METR-LA(207), Traffic(862), ECG5000(140)
- **평가 프로토콜 (VIDA-Style):** 10-seed 학습 × 100-mask 평가 → Mean/Std 보고
- **평가 지표:** RevIN 비활성화, 비정규화 원본 공간에서 ObsMAE / ObsRMSE 계산 (null_val=0.0 제외)
- **기본 설정:** d_model=128, batch_size=64, missing_rate=0.85, loss=mae, temporal=conv1d, fp32
- **K:** 데이터셋별 Elbow 분석으로 결정 (아래 Step 0 참조)
- **주의:** amp_bf16 OFF 통일 (GTX 1080 Ti 호환)

## Step 0. Codebook K 선정 (메인 실험 전 필수)

Stage 1(10 epoch)만 학습하여 Q_full 임베딩을 수집한 후, K-Means Elbow + Silhouette 분석으로 데이터셋별 최적 K를 결정한다. **학습 1회만 필요** (K마다 재학습 불필요).

```bash
# 각 데이터셋별 실행
python scripts/select_k.py --dataset solar --data_dir ./data
python scripts/select_k.py --dataset metr-la --data_dir ./data
python scripts/select_k.py --dataset traffic --data_dir ./data
python scripts/select_k.py --dataset ecg5000 --data_dir ./data

# K 후보 변경 시
python scripts/select_k.py --dataset solar --data_dir ./data --k_candidates 4 8 16 32 64 128
```

**출력:**
- `experiments/results/k_selection_{dataset}.json` — K별 Inertia/Silhouette + 추천 K
- `experiments/results/k_selection_{dataset}.png` — Elbow/Silhouette plot

**결정된 K를 이후 모든 실험에 `--codebook_K` 로 반영한다.**

## 2. RQ1: Main Comparison (Table 1 — 4-Row Bound Comparison)

4개 행으로 구성하여 **두 가지 기여를 정량적으로 분리**한다.

| Row | 표기명 | 세팅 | 의의 |
|-----|--------|------|------|
| 1 | MTGNN (Partial) | 15% 관측 + 85% Zero-fill → 표준 1D MTGNN | VSF 업계 하한선 |
| 2 | MTGNN (Oracle) | 100% 완벽 데이터 → 표준 1D MTGNN | VSF 업계 상한선 |
| 3 | COMET w/o Codebook | 15% 관측 → COMET e2e 백본 (Codebook 제거) | **기여 1: e2e 아키텍처** |
| 4 | **COMET (Ours)** | 15% 관측 → COMET 전체 파이프라인 | **기여 2: + Codebook** |

**스토리라인:**
- Row 2 → 3: e2e 아키텍처만으로 결측 85% 상황에서도 Oracle 초과 → 기여 1 입증
- Row 3 → 4: Codebook 추가로 추가 개선 → 기여 2 입증
- Row 3은 **기여 분해점 (decomposition point)** 역할

**실험 커맨드:**
- Row 1, 2: 표준 MTGNN — 기존 VSF 논문(VIDA 등) 수치 직접 인용. 직접 실험 불필요. (단, 평가 프로토콜 일치 확인: data split, seq/pred_len, 지표 계산 방식)
- Row 3:
```bash
python scripts/train.py --dataset solar --data_dir ./data --no_codebook
```
- Row 4:
```bash
python scripts/train.py --dataset solar --data_dir ./data
```

**10-seed 반복** (아래 seed 가이드 참조), 4개 데이터셋 전부 수행.

## 3. RQ1-b: SOTA 베이스라인 비교 (Main Comparison — Extended)

- 통계/ML 보간: MICE, KNNE, TRMF
- 딥러닝 보간: CSDI, SAITS, SS-GAN
- VSF 특화: FDW, TRF, GIMCC, GinAR, VIDA
- **핵심 어필:** 파라미터 효율성 (COMET ~1.4M)
- 별도 repo에서 실행, 결과만 experiments/results/에 기록

## 4. RQ2: 결측률 강건성 (Missing Rate Sensitivity)

관측 비율별 ObsMAE 곡선 (결측률 50%~95%)

```bash
for mr in 0.50 0.65 0.75 0.85 0.95; do
  python scripts/train.py --dataset solar --data_dir ./data --missing_rate $mr
done
```

## 5. RQ3: 아키텍처 절제 (Ablation Study)

### w/o embedding input (ts_input)
```bash
python scripts/train.py --dataset solar --data_dir ./data --ts_input
```

### w/o Codebook
```bash
python scripts/train.py --dataset solar --data_dir ./data --no_codebook
```

### Temporal path 비교 (기본=conv1d)
```bash
python scripts/train.py --dataset solar --data_dir ./data --temporal_type mamba
python scripts/train.py --dataset solar --data_dir ./data --temporal_type transformer
python scripts/train.py --dataset solar --data_dir ./data --temporal_type identity
```

### Codebook K 민감도
Step 0의 Elbow 결과를 기반으로, 선정된 K 주변 값들의 실제 성능 비교. Solar + Traffic 2개 데이터셋.
```bash
for K in 4 8 16 32 64; do
  python scripts/train.py --dataset solar --data_dir ./data --codebook_K $K
done
```

### Head 교체 (optional)
```bash
python scripts/train.py --dataset solar --data_dir ./data --head_type astgcn
python scripts/train.py --dataset solar --data_dir ./data --head_type mstgcn
python scripts/train.py --dataset solar --data_dir ./data --head_type tgcn
```

## 6. RQ4: 학습 레시피 검증 (Training Strategy)

```bash
# 엔트로피 정규화 비교
python scripts/train.py --dataset solar --data_dir ./data --entropy_reg_weight 0.05
python scripts/train.py --dataset solar --data_dir ./data --entropy_reg_weight 0.2

# Stage 2 안정화 기간
python scripts/train.py --dataset solar --data_dir ./data --stage2_max_epochs 1
python scripts/train.py --dataset solar --data_dir ./data --stage2_max_epochs 5
```

## 7. RQ5: 해석 가능성 (Interpretability)

- t-SNE 시각화: Q_full → 16개 코드북 엔트리 할당
- Solar: 낮/밤 패턴의 클러스터 분리 시각화
- 별도 시각화 스크립트 필요 (scripts/visualize_codebook.py)

## 10-seed 실행 가이드

본실험은 seed를 바꿔가며 10회 반복:
```bash
for seed in 42 123 456 789 1024 2048 3072 4096 5120 6144; do
  python scripts/train.py --dataset solar --data_dir ./data --seed $seed
done
```

평가(100-mask):
```bash
python scripts/evaluate.py logs/<exp_dir> --missing_rate 0.85 --n_samples 100
```
