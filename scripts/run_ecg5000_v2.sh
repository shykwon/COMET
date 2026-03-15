#!/bin/bash
# ECG5000 v2: MTGNN-learned adj, new nocb (Stage A), per-horizon RMSE
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

# MTGNN comet seed 0 checkpoint (adj source)
ADJ_FROM="logs/comet_ecg5000_K16_conv1d_ra0_s0_20260314_101251"

echo "============================================================"
echo "[$(date)] ECG5000 v2 Experiments"
echo "============================================================"

# ============================================================
# Phase 0: Re-eval MTGNN comet (RMSE fix only)
# ============================================================
echo "[$(date)] === Phase 0: MTGNN comet re-eval (RMSE fix) ==="
for dir in logs/comet_ecg5000_K16_conv1d_ra0_s*; do
  if [ -d "$dir" ]; then
    rm -f "$dir/eval_100samples_mr0.85.json"
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64
  fi
done

# ============================================================
# Phase 1: COMET with other heads (MTGNN-learned adj)
# ============================================================
for HEAD in astgcn mstgcn tgcn; do
  echo "[$(date)] === COMET $HEAD training (adj from MTGNN) ==="
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[$(date)] Training COMET $HEAD seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $HEAD --adj_from $ADJ_FROM --amp_bf16
  done

  echo "[$(date)] === COMET $HEAD eval ==="
  for dir in $(ls -d logs/comet_ecg5000_K16_conv1d_${HEAD}_s* 2>/dev/null); do
    if [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
      echo "[$(date)] Eval $dir"
      python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64 --adj_from $ADJ_FROM
    fi
  done
done

# ============================================================
# Phase 2: nocb all heads (Stage A decoder)
# ============================================================
for HEAD in mtgnn astgcn mstgcn tgcn; do
  echo "[$(date)] === nocb $HEAD training ==="
  EXTRA_ARGS=""
  if [ "$HEAD" != "mtgnn" ]; then
    EXTRA_ARGS="--adj_from $ADJ_FROM"
  fi
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[$(date)] Training nocb $HEAD seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $HEAD --no_codebook $EXTRA_ARGS --amp_bf16
  done

  echo "[$(date)] === nocb $HEAD eval ==="
  EVAL_ADJ=""
  if [ "$HEAD" != "mtgnn" ]; then
    EVAL_ADJ="--adj_from $ADJ_FROM"
  fi
  for dir in $(ls -d logs/comet_ecg5000_K16_conv1d_${HEAD}_nocb_s* logs/comet_ecg5000_K16_conv1d_nocb_s* 2>/dev/null); do
    if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
      echo "[$(date)] Eval $dir"
      python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64 $EVAL_ADJ
    fi
  done
done

echo ""
echo "============================================================"
echo "[$(date)] ECG5000 v2 all done!"
echo "============================================================"
