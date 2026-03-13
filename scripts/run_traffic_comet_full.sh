#!/bin/bash
# Traffic COMET Full Only: 4 heads × 10 seeds = 40 runs
# Priority run: get main model results first, ablations later
# K=32, batch_size=16, seeds 0-9
# BS=16 to avoid NVML crash in MIG 20GB (Stage 2 needs forward+forward_full+backward)

export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONUNBUFFERED=1

DATASET="traffic"
DATA_DIR="./data/raw"
K=32
BS=16
SEEDS=(0 1 2 3 4 5 6 7 8 9)
HEADS=("mtgnn" "astgcn" "mstgcn" "tgcn")

TOTAL=$(( ${#HEADS[@]} * ${#SEEDS[@]} ))
COUNT=0

echo "============================================================"
echo "Traffic COMET Full: $TOTAL runs"
echo "K=$K, batch_size=$BS, seeds=${SEEDS[*]}"
echo "Heads: ${HEADS[*]}"
echo "============================================================"

for head in "${HEADS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] COMET | head=$head | seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --head_type $head --amp_bf16
  done
done

echo ""
echo "============================================================"
echo "All $TOTAL COMET full runs completed!"
echo "============================================================"
