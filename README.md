# COMET

**CO**debook-augmented **M**ultivariate time-series forecasting with **E**xpertise **T**ransfer

## Overview

COMET addresses Variable Subset Forecasting (VSF), where an arbitrary subset of sensors may be unavailable at inference time. It maintains a codebook of normal-state system patterns learned during training and uses them to restore missing variate embeddings at inference.

**Architecture (End-to-End):**
1. **Patch Embedding** — Splits time series into patches and projects to d_model dimensions
2. **CI-Mamba** — Per-variate independent temporal encoding via Mamba SSM
3. **Patch-Level Encoder** — Compresses observed variate patches into a system state summary (Q_sub)
4. **Codebook Lookup** — Matches Q_sub to learned normal-state patterns
5. **Two-Stage Decoder** — Restores missing variate patches via observed cross-attention (Stage A) + codebook cross-attention (Stage B)
6. **MTGNN Head** — Receives patch embeddings directly (in_dim=d_model), spatiotemporal graph prediction

**Training** uses a 3-stage curriculum:
- Stage 1: Warm-up with full observation (codebook initialization via K-Means)
- Stage 2: Progressive masking with alignment losses (InfoNCE + KL matching)
- Stage 3: Fine-tuning with lambda cosine decay

## Installation

```bash
pip install -e .

# Optional: for full Mamba support (requires CUDA 12+)
pip install causal-conv1d mamba-ssm
```

## Data Preparation

Place dataset files in the `data/` directory:

```
data/
├── solar.txt          # Solar energy (137 variates)
├── metr-la.h5         # METR-LA traffic speed (207 variates)
├── traffic.txt        # Traffic occupancy (862 variates)
├── electricity.txt    # Electricity consumption (321 variates)
├── ETTh1.csv          # Electricity Transformer Temperature
└── weather.csv        # Weather (21 variates)
```

## Training

```bash
# Solar dataset (default config)
python scripts/train.py --dataset solar --data_dir ./data --batch_size 16 --amp_bf16

# Ablation: time-series input (in_dim=1, like baselines)
python scripts/train.py --dataset solar --data_dir ./data --batch_size 16 --amp_bf16 --ts_input

# Ablation: without codebook
python scripts/train.py --dataset solar --data_dir ./data --batch_size 16 --amp_bf16 --no_codebook
```

## Evaluation

100-times random sensor mask evaluation:

```bash
python scripts/evaluate.py logs/comet_solar_K16_s42_YYYYMMDD_HHMMSS \
  --missing_rate 0.85 --n_samples 100
```

## Citation

```bibtex
@article{comet2025,
  title={COMET: Codebook-augmented Multivariate Time-series Forecasting with Expertise Transfer},
  author={},
  year={2025}
}
```

## License

MIT
