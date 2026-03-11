#!/bin/bash
# ETTm1 Main Experiment: 2 conditions × 4 heads × 10 seeds = 80 runs
# N=7 variates, 15-min, 69680 timesteps
# K=8 (N=7에 적합), batch_size=256, seq_len=12

export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONUNBUFFERED=1

DATASET="ETTm1"
DATA_DIR="./data/raw"
K=8
BS=256
SEQ_LEN=12
PRED_LEN=12
SEEDS=(0 1 2 3 4 5 6 7 8 9)
HEADS=("mtgnn" "astgcn" "mstgcn" "tgcn")

TOTAL=$((2 * ${#HEADS[@]} * ${#SEEDS[@]}))
COUNT=0

echo "============================================================"
echo "ETTm1 Main Experiment: $TOTAL runs"
echo "K=$K, BS=$BS, seq_len=$SEQ_LEN, pred_len=$PRED_LEN"
echo "Seeds: ${SEEDS[*]}"
echo "Heads: ${HEADS[*]}"
echo "============================================================"

for head in "${HEADS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # COMET (full)
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] COMET | head=$head | seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $head --amp_bf16

    # COMET w/o Codebook
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] no_codebook | head=$head | seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $head --no_codebook --amp_bf16
  done
done

echo ""
echo "============================================================"
echo "All $TOTAL runs completed!"
echo "============================================================"
