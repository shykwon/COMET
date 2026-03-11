#!/bin/bash
# ETTm1 K=4 Experiment: COMET only × 4 heads × 10 seeds = 40 runs
# N=7 variates, K=4 (smaller codebook ablation)

export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONUNBUFFERED=1

DATASET="ETTm1"
DATA_DIR="./data/raw"
K=4
BS=256
SEQ_LEN=12
PRED_LEN=12
SEEDS=(0 1 2 3 4 5 6 7 8 9)
HEADS=("mtgnn" "astgcn" "mstgcn" "tgcn")

TOTAL=$(( ${#HEADS[@]} * ${#SEEDS[@]} ))
COUNT=0

echo "============================================================"
echo "ETTm1 K=4 Experiment: $TOTAL runs"
echo "K=$K, BS=$BS, seq_len=$SEQ_LEN, pred_len=$PRED_LEN"
echo "Seeds: ${SEEDS[*]}"
echo "Heads: ${HEADS[*]}"
echo "============================================================"

for head in "${HEADS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] COMET K=$K | head=$head | seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $head --amp_bf16
  done
done

echo ""
echo "============================================================"
echo "All $TOTAL K=4 runs completed!"
echo "============================================================"
