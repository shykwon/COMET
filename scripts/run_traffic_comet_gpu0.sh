#!/bin/bash
# Traffic COMET — GPU 0: MTGNN + ASTGCN, 5 seeds + eval
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

DATASET="traffic"
DATA_DIR="./data/raw"
K=16
BS=32
SEQ_LEN=12
PRED_LEN=12
SEEDS="0 1 2 3 4"

echo "============================================================"
echo "[$(date)] Traffic COMET GPU 0 (MTGNN + ASTGCN)"
echo "============================================================"

# --- MTGNN ---
echo "[$(date)] === MTGNN ==="
for seed in $SEEDS; do
  echo "[$(date)] Training MTGNN seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type mtgnn --amp_bf16
done

echo "[$(date)] MTGNN eval..."
for dir in $(ls -d logs/comet_traffic_K16_conv1d_s* 2>/dev/null | grep -v nocb | grep -v astgcn | grep -v mstgcn | grep -v tgcn); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16
  fi
done

# adj source from MTGNN seed 0
ADJ_FROM=$(ls -d logs/comet_traffic_K16_conv1d_s0_* 2>/dev/null | grep -v nocb | grep -v astgcn | grep -v mstgcn | grep -v tgcn | head -1)
echo "[$(date)] ADJ: $ADJ_FROM"

# --- ASTGCN ---
echo "[$(date)] === ASTGCN ==="
for seed in $SEEDS; do
  echo "[$(date)] Training ASTGCN seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type astgcn --adj_from $ADJ_FROM --amp_bf16
done

echo "[$(date)] ASTGCN eval..."
for dir in $(ls -d logs/comet_traffic_K16_conv1d_astgcn_s* 2>/dev/null | grep -v nocb); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16 --adj_from $ADJ_FROM
  fi
done

echo "[$(date)] GPU 0 done!"
