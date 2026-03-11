#!/bin/bash
# ETTm1 Full Pipeline: Train → 100-mask Eval → Results Summary
# N=7 variates, 15-min, 69680 timesteps
# K=8 (N=7에 적합), batch_size=64, seq_len=12

export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONUNBUFFERED=1

DATASET="ETTm1"
DATA_DIR="./data/raw"
K=8
BS=64
SEQ_LEN=12
PRED_LEN=12
SEEDS=(0 1 2 3 4 5 6 7 8 9)
HEADS=("mtgnn" "astgcn" "mstgcn" "tgcn")

TOTAL=$((2 * ${#HEADS[@]} * ${#SEEDS[@]}))
COUNT=0

# ============================================================
# Stage 1: Training (80 runs)
# ============================================================
echo "============================================================"
echo "ETTm1 [1/3] Training: $TOTAL runs"
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
echo "ETTm1 [1/3] Training complete!"
echo "============================================================"

# ============================================================
# Stage 2: 100-mask Evaluation
# ============================================================
echo ""
echo "============================================================"
echo "ETTm1 [2/3] 100-mask Evaluation"
echo "============================================================"

bash scripts/eval_all.sh --dataset ETTm1

echo ""
echo "============================================================"
echo "ETTm1 [2/3] Evaluation complete!"
echo "============================================================"

# ============================================================
# Stage 3: Results Summary
# ============================================================
echo ""
echo "============================================================"
echo "ETTm1 [3/3] Results Summary"
echo "============================================================"

python3 -c "
import json, os, yaml
from pathlib import Path
from collections import defaultdict
import numpy as np

results = defaultdict(lambda: defaultdict(list))

for d in sorted(Path('logs').glob('comet_ETTm1_*')):
    eval_file = d / 'eval_100samples_mr0.85.json'
    config_file = d / 'config.yaml'
    if not eval_file.exists() or not config_file.exists():
        continue

    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    with open(eval_file) as f:
        ev = json.load(f)

    head = cfg['model'].get('head_type', 'mtgnn')
    use_cb = cfg['model'].get('use_codebook', True)
    condition = 'comet' if use_cb else 'nocb'
    key = f'{head}_{condition}'

    results[key]['ObsMAE'].append(ev['ObsMAE_mean'])
    results[key]['ObsRMSE'].append(ev['ObsRMSE_mean'])

print(f\"\"\"
{'='*70}
ETTm1 Results Summary (100-mask, 10-seed)
{'='*70}
{'Method':<25} {'ObsMAE':>15} {'ObsRMSE':>15} {'n_seeds':>10}
{'-'*70}\"\"\")

# Sort: comet first, then nocb
for key in sorted(results.keys(), key=lambda k: ('1' if 'nocb' in k else '0') + k):
    maes = results[key]['ObsMAE']
    rmses = results[key]['ObsRMSE']
    n = len(maes)
    print(f'{key:<25} {np.mean(maes):>7.4f}±{np.std(maes):.4f} {np.mean(rmses):>7.4f}±{np.std(rmses):.4f} {n:>10}')

# Save to experiments/results
output = {
    'description': 'RQ1 ETTm1 4-row bound comparison (Row 3 & 4)',
    'dataset': 'ETTm1',
    'protocol': '10-seed x 100-mask',
    'runs': {}
}
for key in sorted(results.keys()):
    maes = results[key]['ObsMAE']
    rmses = results[key]['ObsRMSE']
    output['runs'][key] = {
        'n_seeds': len(maes),
        'test': {
            'ObsMAE': {'mean': round(float(np.mean(maes)), 4), 'std': round(float(np.std(maes)), 4)},
            'ObsRMSE': {'mean': round(float(np.mean(rmses)), 4), 'std': round(float(np.std(rmses)), 4)},
        }
    }

os.makedirs('experiments/results', exist_ok=True)
with open('experiments/results/RQ1_ETTm1.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f\"\"\"\n{'='*70}\nSaved to experiments/results/RQ1_ETTm1.json\n{'='*70}\"\"\")
"

echo ""
echo "============================================================"
echo "ETTm1 Full Pipeline Complete!"
echo "============================================================"
