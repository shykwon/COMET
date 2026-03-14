#!/bin/bash
# Solar 1안/2안 비교: 2안 대기 → 평가 → nocb 실험
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTHONUNBUFFERED=1

DATASET="solar"
DATA_DIR="./data/raw"
K=16
BS=64
SEQ_LEN=12
PRED_LEN=12
SEED=0

# 이미 완료된 디렉토리
GATING_DIR="logs/comet_solar_K16_conv1d_ra0_s0_20260313_093609"
LEARN_DIR="logs/comet_solar_K16_conv1d_ra0_s0_20260313_121645"

# ============================================================
# Step 1: Wait for 2안 learnable training
# ============================================================
LEARN_PID=$(pgrep -f "train.py.*solar")
if [ -n "$LEARN_PID" ]; then
  echo "[$(date)] Waiting for 2안 learnable (PID: $LEARN_PID)..."
  while kill -0 "$LEARN_PID" 2>/dev/null; do
    sleep 30
  done
fi
echo "[$(date)] 2안 training done."

# ============================================================
# Step 2: 100-mask eval (1안 + 2안)
# ============================================================
echo "[$(date)] Eval 1안 (gating)..."
git checkout exp/codebook-gating
python3 scripts/evaluate.py "$GATING_DIR" --missing_rate 0.85 --n_samples 100 --batch_size 64 --cpu

echo "[$(date)] Eval 2안 (learnable)..."
git checkout exp/codebook-learnable
python3 scripts/evaluate.py "$LEARN_DIR" --missing_rate 0.85 --n_samples 100 --batch_size 64 --cpu

# ============================================================
# Step 3: 1안 nocb
# ============================================================
echo "[$(date)] Training 1안 nocb..."
git checkout exp/codebook-gating
python3 scripts/train.py \
  --dataset $DATASET --data_dir $DATA_DIR \
  --codebook_K $K --batch_size $BS --seed $SEED \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN \
  --head_type mtgnn --no_codebook --restore_alpha 0 --amp_bf16

GATING_NOCB=$(ls -dt logs/comet_solar_K16_conv1d_nocb*s0_* 2>/dev/null | head -1)
echo "[$(date)] Eval 1안 nocb..."
python3 scripts/evaluate.py "$GATING_NOCB" --missing_rate 0.85 --n_samples 100 --batch_size 64 --cpu

# ============================================================
# Step 4: 2안 nocb
# ============================================================
echo "[$(date)] Training 2안 nocb..."
git checkout exp/codebook-learnable
python3 scripts/train.py \
  --dataset $DATASET --data_dir $DATA_DIR \
  --codebook_K $K --batch_size $BS --seed $SEED \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN \
  --head_type mtgnn --no_codebook --restore_alpha 0 --amp_bf16

LEARN_NOCB=$(ls -dt logs/comet_solar_K16_conv1d_nocb*s0_* 2>/dev/null | head -1)
echo "[$(date)] Eval 2안 nocb..."
python3 scripts/evaluate.py "$LEARN_NOCB" --missing_rate 0.85 --n_samples 100 --batch_size 64 --cpu

echo ""
echo "============================================================"
echo "[$(date)] Solar comparison all done!"
echo "============================================================"
