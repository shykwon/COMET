#!/bin/bash
# Traffic: mtgnn only, comet + nocb, 10 seeds + 100-mask eval
set -e
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTHONUNBUFFERED=1
cd /home/elicer/COMET

DATASET="traffic"
DATA_DIR="./data/raw"
K=16
BS=32
SEQ_LEN=12
PRED_LEN=12
HEAD="mtgnn"

echo "============================================================"
echo "[$(date)] Traffic mtgnn Experiments (40 runs)"
echo "============================================================"

# --- COMET (gating) 10 seeds ---
echo "[$(date)] === COMET mtgnn training ==="
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date)] Training COMET seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type $HEAD --restore_alpha 0 --amp_bf16
done

echo "[$(date)] === COMET mtgnn eval ==="
for dir in $(ls -d logs/comet_traffic_* 2>/dev/null | grep -v nocb | grep "_s[0-9]_"); do
  if [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32
  fi
done

# --- nocb 10 seeds ---
echo "[$(date)] === nocb mtgnn training ==="
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date)] Training nocb seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type $HEAD --no_codebook --restore_alpha 0 --amp_bf16
done

echo "[$(date)] === nocb mtgnn eval ==="
for dir in $(ls -d logs/comet_traffic_*nocb* 2>/dev/null | grep "_s[0-9]_"); do
  if [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32
  fi
done

echo ""
echo "============================================================"
echo "[$(date)] Traffic all done!"
echo "============================================================"
