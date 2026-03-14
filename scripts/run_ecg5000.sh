#!/bin/bash
# ECG5000: COMET first (4 heads × 10 seeds + eval), then nocb
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

echo "============================================================"
echo "[$(date)] ECG5000 Experiments"
echo "============================================================"

# ============================================================
# Phase 1: COMET (gating) — 4 heads × 10 seeds + eval
# ============================================================
for HEAD in mtgnn astgcn mstgcn tgcn; do
  echo "[$(date)] === COMET $HEAD training ==="
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[$(date)] Training COMET $HEAD seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $HEAD --restore_alpha 0 --amp_bf16
  done

  echo "[$(date)] === COMET $HEAD eval ==="
  for dir in $(ls -d logs/comet_ecg5000_* 2>/dev/null | grep -v nocb | grep "_s[0-9]_"); do
    if [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
      echo "[$(date)] Eval $dir"
      python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64
    fi
  done
done

echo ""
echo "============================================================"
echo "[$(date)] Phase 1 (COMET) done! Starting Phase 2 (nocb)..."
echo "============================================================"

# ============================================================
# Phase 2: nocb — 4 heads × 10 seeds + eval
# ============================================================
for HEAD in mtgnn astgcn mstgcn tgcn; do
  echo "[$(date)] === nocb $HEAD training ==="
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[$(date)] Training nocb $HEAD seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $HEAD --no_codebook --restore_alpha 0 --amp_bf16
  done

  echo "[$(date)] === nocb $HEAD eval ==="
  for dir in $(ls -d logs/comet_ecg5000_*nocb* 2>/dev/null | grep "_s[0-9]_"); do
    if [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
      echo "[$(date)] Eval $dir"
      python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 64
    fi
  done
done

echo ""
echo "============================================================"
echo "[$(date)] ECG5000 all done!"
echo "============================================================"
