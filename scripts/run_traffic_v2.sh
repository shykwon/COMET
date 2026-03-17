#!/bin/bash
# Traffic: mtgnn only, comet + nocb, 10 seeds + 100-mask eval, bs=16
set -e
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTHONUNBUFFERED=1
cd /home/elicer/COMET

DATASET="traffic"
DATA_DIR="./data/raw"
K=16
BS=16
SEQ_LEN=12
PRED_LEN=12
HEAD="mtgnn"

echo "============================================================"
echo "[$(date)] Traffic mtgnn Experiments (bs=16)"
echo "============================================================"

# --- COMET (gating) 10 seeds ---
echo "[$(date)] === COMET mtgnn training ==="
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date)] Training COMET seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type $HEAD --amp_bf16
done

echo "[$(date)] === COMET mtgnn eval ==="
for dir in $(ls -d logs/comet_traffic_K16_conv1d_s* 2>/dev/null | grep -v nocb); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 16
  fi
done

# --- nocb 10 seeds (skip for now) ---
# echo "[$(date)] === nocb mtgnn training ==="
# for seed in 0 1 2 3 4 5 6 7 8 9; do
#   python3 scripts/train.py --dataset $DATASET --data_dir $DATA_DIR \
#     --codebook_K $K --batch_size $BS --seed $seed \
#     --seq_len $SEQ_LEN --pred_len $PRED_LEN \
#     --head_type $HEAD --no_codebook --amp_bf16
# done

echo ""
echo "============================================================"
echo "[$(date)] Traffic all done!"
echo "============================================================"
