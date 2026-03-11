#!/bin/bash
# Solar 3-GPU Parallel: 80 runs across 3 Titan X GPUs
# N=137, K=16, BS=32, fp32 (no bf16)
# Distribution: 26 + 28 + 26 = 80 runs

export PYTHONUNBUFFERED=1
mkdir -p logs

DATASET="solar"
DATA_DIR="./data/raw"
K=16
BS=32
SEQ_LEN=96
PRED_LEN=12
SEEDS=(0 1 2 3 4 5 6 7 8 9)

run_one() {
  local GPU=$1 HEAD=$2 SEED=$3 NOCB=$4
  local EXTRA=""
  local TAG="COMET"
  if [[ "$NOCB" == "1" ]]; then
    EXTRA="--no_codebook"
    TAG="no_codebook"
  fi
  echo "[GPU $GPU] $TAG | head=$HEAD | seed=$SEED"
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $SEED \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type $HEAD $EXTRA
}

gpu0_worker() {
  # mtgnn 10 seeds × 2 = 20, tgcn seeds 0-2 × 2 = 6 → total 26
  for seed in "${SEEDS[@]}"; do
    run_one 0 mtgnn $seed 0
    run_one 0 mtgnn $seed 1
  done
  for seed in 0 1 2; do
    run_one 0 tgcn $seed 0
    run_one 0 tgcn $seed 1
  done
}

gpu1_worker() {
  # astgcn 10 seeds × 2 = 20, tgcn seeds 3-6 × 2 = 8 → total 28
  for seed in "${SEEDS[@]}"; do
    run_one 1 astgcn $seed 0
    run_one 1 astgcn $seed 1
  done
  for seed in 3 4 5 6; do
    run_one 1 tgcn $seed 0
    run_one 1 tgcn $seed 1
  done
}

gpu2_worker() {
  # mstgcn 10 seeds × 2 = 20, tgcn seeds 7-9 × 2 = 6 → total 26
  for seed in "${SEEDS[@]}"; do
    run_one 2 mstgcn $seed 0
    run_one 2 mstgcn $seed 1
  done
  for seed in 7 8 9; do
    run_one 2 tgcn $seed 0
    run_one 2 tgcn $seed 1
  done
}

echo "============================================================"
echo "Solar 3-GPU Experiment: 80 runs"
echo "K=$K, BS=$BS, seq_len=$SEQ_LEN, pred_len=$PRED_LEN"
echo "GPU 0: mtgnn(20) + tgcn s0-2(6) = 26 runs"
echo "GPU 1: astgcn(20) + tgcn s3-6(8) = 28 runs"
echo "GPU 2: mstgcn(20) + tgcn s7-9(6) = 26 runs"
echo "============================================================"

gpu0_worker > logs/solar_gpu0.log 2>&1 &
echo "GPU 0 started: PID $!"

gpu1_worker > logs/solar_gpu1.log 2>&1 &
echo "GPU 1 started: PID $!"

gpu2_worker > logs/solar_gpu2.log 2>&1 &
echo "GPU 2 started: PID $!"

wait
echo ""
echo "============================================================"
echo "All 80 runs completed!"
echo "============================================================"
