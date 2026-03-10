#!/bin/bash
# Solar Main Experiment: 2 conditions × 4 heads × 10 seeds = 80 runs
# K=16, batch_size=64, amp_bf16 ON, seeds 0-9

DATASET="solar"
DATA_DIR="./data/raw"
K=16
BS=64
SEEDS=(0 1 2 3 4 5 6 7 8 9)
HEADS=("mtgnn" "astgcn" "mstgcn" "tgcn")

TOTAL=$((2 * ${#HEADS[@]} * ${#SEEDS[@]}))
COUNT=0

echo "============================================================"
echo "Solar Main Experiment: $TOTAL runs"
echo "K=$K, batch_size=$BS, amp_bf16=ON, seeds=${SEEDS[*]}"
echo "============================================================"

for head in "${HEADS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # Row 4: COMET (full)
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] COMET | head=$head | seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --head_type $head --amp_bf16

    # Row 3: COMET w/o Codebook
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] no_codebook | head=$head | seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --head_type $head --no_codebook --amp_bf16
  done
done

echo ""
echo "============================================================"
echo "All $TOTAL runs completed!"
echo "============================================================"
