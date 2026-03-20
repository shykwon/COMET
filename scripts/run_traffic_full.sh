#!/bin/bash
# Traffic Full: 4 heads × comet/nocb × 5 seeds + 100-mask eval
# Target: A30 GPU (24GB), bs=32 가능
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
SEEDS="0 1 2 3 4"

# MTGNN comet seed 0에서 adj 추출 (다른 head용)
# 먼저 MTGNN을 돌려야 adj를 뽑을 수 있음
ADJ_FROM=""

echo "============================================================"
echo "[$(date)] Traffic Full Experiments (A30 GPU)"
echo "  4 heads × comet/nocb × 5 seeds + eval"
echo "============================================================"

# ============================================================
# Phase 1: MTGNN (자체 learned adj)
# ============================================================
echo "[$(date)] === Phase 1: MTGNN ==="

echo "[$(date)] COMET MTGNN training..."
for seed in $SEEDS; do
  echo "[$(date)] Training COMET MTGNN seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type mtgnn --amp_bf16
done

echo "[$(date)] COMET MTGNN eval..."
for dir in $(ls -d logs/comet_traffic_K16_conv1d_s* 2>/dev/null | grep -v nocb | grep -v astgcn | grep -v mstgcn | grep -v tgcn); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16
  fi
done

echo "[$(date)] nocb MTGNN training..."
for seed in $SEEDS; do
  echo "[$(date)] Training nocb MTGNN seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type mtgnn --no_codebook --amp_bf16
done

echo "[$(date)] nocb MTGNN eval..."
for dir in $(ls -d logs/comet_traffic_K16_conv1d_nocb_s* 2>/dev/null); do
  if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
    echo "[$(date)] Eval $dir"
    python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16
  fi
done

# adj 추출 소스: MTGNN comet seed 0
ADJ_FROM=$(ls -d logs/comet_traffic_K16_conv1d_s0_* 2>/dev/null | grep -v nocb | grep -v astgcn | grep -v mstgcn | grep -v tgcn | head -1)
echo "[$(date)] ADJ source: $ADJ_FROM"

# ============================================================
# Phase 2: ASTGCN, MSTGCN, TGCN (MTGNN-learned adj)
# ============================================================
for HEAD in astgcn mstgcn tgcn; do
  echo "[$(date)] === Phase 2: $HEAD ==="

  echo "[$(date)] COMET $HEAD training..."
  for seed in $SEEDS; do
    echo "[$(date)] Training COMET $HEAD seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $HEAD --adj_from $ADJ_FROM --amp_bf16
  done

  echo "[$(date)] COMET $HEAD eval..."
  for dir in $(ls -d logs/comet_traffic_K16_conv1d_${HEAD}_s* 2>/dev/null | grep -v nocb); do
    if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
      echo "[$(date)] Eval $dir"
      python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16 --adj_from $ADJ_FROM
    fi
  done

  echo "[$(date)] nocb $HEAD training..."
  for seed in $SEEDS; do
    echo "[$(date)] Training nocb $HEAD seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $HEAD --no_codebook --adj_from $ADJ_FROM --amp_bf16
  done

  echo "[$(date)] nocb $HEAD eval..."
  for dir in $(ls -d logs/comet_traffic_K16_conv1d_${HEAD}_nocb_s* 2>/dev/null); do
    if [ -d "$dir" ] && [ ! -f "$dir/eval_100samples_mr0.85.json" ]; then
      echo "[$(date)] Eval $dir"
      python3 scripts/evaluate.py "$dir" --missing_rate 0.85 --n_samples 100 --batch_size 32 --amp_bf16 --adj_from $ADJ_FROM
    fi
  done
done

echo ""
echo "============================================================"
echo "[$(date)] Traffic Full all done!"
echo "  Total: 4 heads × 2 (comet+nocb) × 5 seeds = 40 trains + 40 evals"
echo "============================================================"
