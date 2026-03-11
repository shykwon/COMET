#!/bin/bash
# Solar Main Experiment: 2 conditions × 4 heads × 10 seeds = 80 runs
# N=137 variates, 52560 timesteps
# Target: Titan X 12GB (fp32, no bf16)

export PYTHONUNBUFFERED=1

DATASET="solar"
DATA_DIR="./data/raw"
K=16
BS=32
SEQ_LEN=12
PRED_LEN=12
SEEDS=(0 1 2 3 4 5 6 7 8 9)
HEADS=("mtgnn" "astgcn" "mstgcn" "tgcn")

TOTAL=$((2 * ${#HEADS[@]} * ${#SEEDS[@]}))
COUNT=0

echo "============================================================"
echo "Solar Main Experiment: $TOTAL runs"
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
      --head_type $head

    # COMET w/o Codebook
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] no_codebook | head=$head | seed=$seed"
    python3 scripts/train.py \
      --dataset $DATASET --data_dir $DATA_DIR \
      --codebook_K $K --batch_size $BS --seed $seed \
      --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --head_type $head --no_codebook
  done
done

echo ""
echo "============================================================"
echo "All $TOTAL runs completed!"
echo "============================================================"
