# COMET 실험 설계

## 1. 기본 실험 환경 (Experimental Setup)

- **데이터셋 (4종):** Solar(137), METR-LA(207), Traffic(862), ECG5000(140)
- **평가 프로토콜 (VIDA-Style):** 10-seed 학습 × 100-mask 평가 → Mean/Std 보고
- **평가 지표:** RevIN 비활성화, 비정규화 원본 공간에서 ObsMAE / ObsRMSE 계산 (null_val=0.0 제외)
- **기본 설정:** d_model=128, K=16, batch_size=64, missing_rate=0.85, loss=mae, temporal=conv1d, fp32
- **주의:** amp_bf16 OFF 통일 (GTX 1080 Ti 호환)

## 2. RQ1-a: VSF 바운드 비교 (Partial vs Oracle)

MTGNN 백본 기준: Partial(0-fill) vs Oracle(100%) vs COMET
- Δ_subset: Partial 대비 향상률
- Δ_complete: Oracle 대비 향상률 (Oracle 초과 강조)

```bash
# Oracle (missing_rate=0)
python scripts/train.py --dataset solar --data_dir ./data --missing_rate 0.0

# Partial (no codebook, 0-fill)
python scripts/train.py --dataset solar --data_dir ./data --no_codebook

# COMET (기본)
python scripts/train.py --dataset solar --data_dir ./data
```

## 3. RQ1-b: SOTA 베이스라인 비교 (Main Comparison)

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
