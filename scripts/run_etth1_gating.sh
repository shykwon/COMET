#!/bin/bash
# ETTh1 gating 10-seed × {comet, nocb} × {mtgnn} + 100-mask eval = 80 runs
# comet 10 train + 10 eval + nocb 10 train + 10 eval = 40+40 = 80
set -e
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTHONUNBUFFERED=1
cd /home/elicer/COMET

DATASET="ETTh1"
DATA_DIR="./data/raw"
K=8
BS=64
SEQ_LEN=12
PRED_LEN=12
HEAD="mtgnn"

echo "============================================================"
echo "[$(date)] ETTh1 Gating Experiments (80 runs)"
echo "============================================================"

# --- COMET (gating) 10 seeds ---
echo "[$(date)] === COMET (gating) training ==="
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date)] Training COMET seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type $HEAD --restore_alpha 0 --amp_bf16
done

echo "[$(date)] === COMET (gating) eval ==="
for dir in logs/comet_ETTh1_K8_conv1d_s*; do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64
  fi
done

# --- nocb 10 seeds ---
echo "[$(date)] === nocb training ==="
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date)] Training nocb seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type $HEAD --no_codebook --restore_alpha 0 --amp_bf16
done

echo "[$(date)] === nocb eval ==="
for dir in logs/comet_ETTh1_K8_conv1d_nocb*s*; do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64
  fi
done

echo ""
echo "============================================================"
echo "[$(date)] ETTh1 all done!"
echo "============================================================"
