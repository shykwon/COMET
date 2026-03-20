#!/bin/bash
# Traffic COMET — GPU 1: MSTGCN + TGCN, 5 seeds + eval
# NOTE: GPU 0에서 MTGNN seed 0 학습이 끝나야 adj를 뽑을 수 있음
# MTGNN seed 0 완료 대기 후 시작
set -e
export CUDA_VISIBLE_DEVICES=1
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
echo "[$(date)] Traffic COMET GPU 1 (MSTGCN + TGCN)"
echo "  Waiting for MTGNN seed 0 on GPU 0..."
echo "============================================================"

# Wait for MTGNN seed 0 to finish (need adj)
while [ ! -d logs/comet_traffic_K16_conv1d_s0_*/best_model.pt ] 2>/dev/null; do
  ADJ_CHECK=$(ls -d logs/comet_traffic_K16_conv1d_s0_* 2>/dev/null | grep -v nocb | grep -v astgcn | grep -v mstgcn | grep -v tgcn | head -1)
  if [ -n "$ADJ_CHECK" ] && [ -f "$ADJ_CHECK/best_model.pt" ]; then
    break
  fi
  echo "[$(date)] Waiting for MTGNN seed 0..."
  sleep 60
done

ADJ_FROM=$(ls -d logs/comet_traffic_K16_conv1d_s0_* 2>/dev/null | grep -v nocb | grep -v astgcn | grep -v mstgcn | grep -v tgcn | head -1)
echo "[$(date)] ADJ: $ADJ_FROM"

# --- MSTGCN ---
echo "[$(date)] === MSTGCN ==="
for seed in $SEEDS; do
  echo "[$(date)] Training MSTGCN seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type mstgcn --adj_from $ADJ_FROM --amp_bf16
done

echo "[$(date)] MSTGCN eval..."
for dir in $(ls -d logs/comet_traffic_K16_conv1d_mstgcn_s* 2>/dev/null | grep -v nocb); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16 --adj_from $ADJ_FROM
  fi
done

# --- TGCN ---
echo "[$(date)] === TGCN ==="
for seed in $SEEDS; do
  echo "[$(date)] Training TGCN seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type tgcn --adj_from $ADJ_FROM --amp_bf16
done

echo "[$(date)] TGCN eval..."
for dir in $(ls -d logs/comet_traffic_K16_conv1d_tgcn_s* 2>/dev/null | grep -v nocb); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16 --adj_from $ADJ_FROM
  fi
done

echo "[$(date)] GPU 1 done!"
