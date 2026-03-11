#!/bin/bash
# Batch 100-mask evaluation for all completed experiments
# Scans logs/ for best_model.pt and runs evaluate.py on each
#
# Usage:
#   bash scripts/eval_all.sh                          # all completed runs
#   bash scripts/eval_all.sh --dataset ETTh1          # filter by dataset
#   bash scripts/eval_all.sh --pattern "K16.*mtgnn"   # filter by regex

export PYTHONUNBUFFERED=1

N_SAMPLES=100
MISSING_RATE=0.85
BATCH_SIZE=64
DATASET_FILTER=""
PATTERN_FILTER=""
CPU_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset) DATASET_FILTER="$2"; shift 2 ;;
    --pattern) PATTERN_FILTER="$2"; shift 2 ;;
    --n_samples) N_SAMPLES="$2"; shift 2 ;;
    --missing_rate) MISSING_RATE="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --cpu) CPU_FLAG="--cpu"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "Batch 100-mask Evaluation"
echo "n_samples=$N_SAMPLES, missing_rate=$MISSING_RATE"
echo "============================================================"

TOTAL=0
DONE=0
SKIP=0
FAIL=0

# Collect eligible dirs
DIRS=()
for d in logs/comet_*/; do
  [ -d "$d" ] || continue
  [ -f "$d/best_model.pt" ] || continue

  name=$(basename "$d")

  # Dataset filter
  if [ -n "$DATASET_FILTER" ]; then
    grep -qi "$DATASET_FILTER" "$d/config.yaml" 2>/dev/null || continue
  fi

  # Pattern filter
  if [ -n "$PATTERN_FILTER" ]; then
    echo "$name" | grep -qE "$PATTERN_FILTER" || continue
  fi

  DIRS+=("$d")
done

TOTAL=${#DIRS[@]}
echo "Found $TOTAL experiments to evaluate"
echo ""

for d in "${DIRS[@]}"; do
  name=$(basename "$d")
  eval_file="$d/eval_${N_SAMPLES}samples_mr${MISSING_RATE}.json"

  # Skip if already evaluated
  if [ -f "$eval_file" ]; then
    SKIP=$((SKIP + 1))
    echo "[SKIP] $name (already evaluated)"
    continue
  fi

  echo "[EVAL] $name"
  python3 scripts/evaluate.py "$d" \
    --missing_rate $MISSING_RATE \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE $CPU_FLAG

  if [ $? -eq 0 ]; then
    DONE=$((DONE + 1))
  else
    FAIL=$((FAIL + 1))
    echo "[FAIL] $name"
  fi
  echo ""
done

echo "============================================================"
echo "Evaluation complete: $DONE done, $SKIP skipped, $FAIL failed (of $TOTAL)"
echo "============================================================"
