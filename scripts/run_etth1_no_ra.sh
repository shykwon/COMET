#!/bin/bash
# ETTh1 Ablation: COMET mtgnn with restore_alpha=0 (no graph masking)
# 10 seeds, codebook ON, but obs_mask not passed to head

export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONUNBUFFERED=1

DATASET="ETTh1"
DATA_DIR="./data/raw"
K=8
BS=64
SEQ_LEN=12
PRED_LEN=12
SEEDS=(0 1 2 3 4 5 6 7 8 9)
TOTAL=${#SEEDS[@]}
COUNT=0

# ============================================================
# Stage 1: Training (10 runs)
# ============================================================
echo "============================================================"
echo "ETTh1 Ablation: mtgnn + restore_alpha=0 (no graph masking)"
echo "$TOTAL runs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
  COUNT=$((COUNT + 1))
  echo ""
  echo "[$COUNT/$TOTAL] COMET (no graph mask) | seed=$seed"
  python3 scripts/train.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --codebook_K $K --batch_size $BS --seed $seed \
    --seq_len $SEQ_LEN --pred_len $PRED_LEN \
    --head_type mtgnn --restore_alpha 0 --amp_bf16
done

echo ""
echo "============================================================"
echo "Training complete! Starting 100-mask evaluation..."
echo "============================================================"

# ============================================================
# Stage 2: 100-mask Evaluation (CPU)
# ============================================================
bash scripts/eval_all.sh --dataset ETTh1 --pattern "ra0" --cpu

echo ""
echo "============================================================"
echo "ETTh1 restore_alpha=0 ablation complete!"
echo "============================================================"
