#!/bin/bash
# ECG5000 v2 resume: continue from nocb astgcn seed 3
set -e
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTHONUNBUFFERED=1
cd /home/elicer/COMET

DATASET="ecg5000"
DATA_DIR="./data/raw"
K=16
BS=64
SEQ_LEN=12
PRED_LEN=12
ADJ_FROM="logs/comet_ecg5000_K16_conv1d_ra0_s0_20260314_101251"

echo "============================================================"
echo "[$(date)] ECG5000 v2 Resume"
echo "============================================================"

# --- nocb astgcn remaining seeds ---
echo "[$(date)] === nocb astgcn training (resume from seed 3) ==="
for seed in 3 4 5 6 7 8 9; do
  DIR_EXISTS=$(ls -d logs/comet_ecg5000_K16_conv1d_astgcn_nocb_s${seed}_* 2>/dev/null | head -1)
  if [ -n "$DIR_EXISTS" ] && [ -f "$DIR_EXISTS/results.json" ]; then
    echo "[$(date)] Skipping astgcn nocb seed=$seed (already done)"
    continue
  fi
  echo "[$(date)] Training nocb astgcn seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type astgcn --no_codebook --adj_from $ADJ_FROM --amp_bf16
done

echo "[$(date)] === nocb astgcn eval ==="
for dir in $(ls -d logs/comet_ecg5000_K16_conv1d_astgcn_nocb_s* 2>/dev/null); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64 --adj_from $ADJ_FROM
  fi
done

# --- nocb mstgcn ---
echo "[$(date)] === nocb mstgcn training ==="
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date)] Training nocb mstgcn seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type mstgcn --no_codebook --adj_from $ADJ_FROM --amp_bf16
done

echo "[$(date)] === nocb mstgcn eval ==="
for dir in $(ls -d logs/comet_ecg5000_K16_conv1d_mstgcn_nocb_s* 2>/dev/null); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64 --adj_from $ADJ_FROM
  fi
done

# --- nocb tgcn ---
echo "[$(date)] === nocb tgcn training ==="
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date)] Training nocb tgcn seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type tgcn --no_codebook --adj_from $ADJ_FROM --amp_bf16
done

echo "[$(date)] === nocb tgcn eval ==="
for dir in $(ls -d logs/comet_ecg5000_K16_conv1d_tgcn_nocb_s* 2>/dev/null); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64 --adj_from $ADJ_FROM
  fi
done

echo ""
echo "============================================================"
echo "[$(date)] ECG5000 v2 resume all done!"
echo "============================================================"
