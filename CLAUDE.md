# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMET (COdebook-augmented Multivariate time-series forecasting with Expertise Transfer) — addresses Variable Subset Forecasting (VSF), where arbitrary sensor subsets may be unavailable at inference time.

## Commands

```bash
# Install
pip install -e .
pip install causal-conv1d mamba-ssm  # Optional, CUDA 12+

# Train
python scripts/train.py --dataset solar --data_dir ./data --batch_size 16 --amp_bf16
python scripts/train.py --dataset solar --data_dir ./data --batch_size 16 --amp_bf16 --ts_input      # ablation: time-series input
python scripts/train.py --dataset solar --data_dir ./data --batch_size 16 --amp_bf16 --no_codebook   # ablation: no codebook

# Evaluate (100-mask)
python scripts/evaluate.py logs/comet_solar_K16_s42_YYYYMMDD_HHMMSS --missing_rate 0.85 --n_samples 100

# CLI overrides (all config keys)
python scripts/train.py --dataset electricity --d_model 256 --batch_size 32 --codebook_K 32 --lr 5e-4
```

No test framework configured. `tests/` exists but is empty.

## Architecture (Forward Pass)

```
x_full [B, N, T] + obs_mask [B, N]
  → PatchEmbedding          [B, N, L, D]     patch_embedding.py
  → CI-Mamba (per-variate)   [B, N, L, D]     temporal.py        (Mamba2 or SimplifiedSSM fallback)
  → PatchLevelEncoder        Q_sub [B, D]     encoder.py         (TransformerEncoder + CLS token)
  → Codebook soft lookup     z_ctx, w, conf   codebook.py        (K=16, tau=0.5, EMA + K-Means init)
  → TwoStageDecoder          [B, N, L, D]     decoder.py         (StageA: obs cross-attn, StageB: codebook cross-attn)
  → MTGNNHead                [B, N, pred_len] forecast_head.py   (dilated inception + MixProp GCN)
```

Main model class: `src/comet/models/comet.py`
- `forward(x_full, obs_mask)` — student path (masked variates)
- `forward_full(x)` — teacher path (all variates observed)

## 3-Stage Curriculum Training

Managed by `CurriculumScheduler` in `src/comet/training/curriculum.py`:

- **Stage 1** (epochs 1–10): mask_ratio=0, lambda=0, task loss only. Collects Q_full for codebook K-Means init at end.
- **Stage 2** (up to 5 epochs): Linear ramp of mask_ratio and alignment lambdas. Transitions when cos_sim ≥ 0.85 for 3 consecutive epochs.
- **Stage 3**: mask_ratio locked at max (0.85), lambdas decay via cosine annealing.

## Loss

`L = L_task + λ_align·InfoNCE + λ_match·KL(w_sub||w_full) + γ·EntropyReg`

All in `src/comet/training/losses.py`. Task loss computed in denormalized space with null_val=0.0 masking.

## Key Design Decisions

- **Dual forward**: Teacher (`forward_full`) runs only when alignment losses are needed (Stage 2+), with `stop_gradient` on teacher outputs.
- **ts_input ablation**: Projects patch embeddings back to time-series `[B,N,T]`, overwrites observed with ground truth, then feeds to MTGNN. Enables fair comparison with time-series-only baselines.
- **Confidence-weighted graph masking**: MTGNNHead scales graph influence of restored variates by `restore_alpha * confidence`. Adaptive alpha enabled by default.
- **Codebook dead entry revival**: Entries with usage < 0.01 are replaced with noisy samples from daytime pool.
- **Null value handling**: Solar/weather nighttime zeros excluded from loss via explicit masking.

## Tensor Conventions

- Time series: `[B, N, T]` — batch, variates, timesteps
- Patches: `[B, N, L, D]` — batch, variates, num_patches, d_model
- Masks: boolean, `True = observed/valid`
- Variable-length obs/miss sets padded to batch max with flag tensors

## Config

`configs/default.yaml` — all hyperparameters. CLI args override any config key.
