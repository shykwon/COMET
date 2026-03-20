#!/bin/bash
# Codebook Ablation: Solar seed 0, MTGNN, 100-mask eval
# A: hard lookup, B: no revival, C: no EMA, E: no entropy, F: B+E
set -e
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

DATASET="solar"
DATA_DIR="./data/raw"
K=16
BS=64
SEED=0
COMMON="--dataset $DATASET --data_dir $DATA_DIR --codebook_K $K --batch_size $BS --seed $SEED --seq_len 12 --pred_len 12 --head_type mtgnn --amp_bf16"

echo "============================================================"
echo "[$(date)] Codebook Ablation (Solar, seed 0)"
echo "============================================================"

# --- Baseline (이미 있으면 스킵) ---
echo "[$(date)] === Baseline (default) ==="
python3 scripts/train.py $COMMON
BASELINE_DIR=$(ls -dt logs/comet_solar_K16_conv1d_s0_* 2>/dev/null | grep -v nocb | grep -v hardlk | grep -v norev | grep -v noema | grep -v noent | head -1)
if [ -n "$BASELINE_DIR" ] && [ ! -f "$BASELINE_DIR/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$BASELINE_DIR" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- A: Hard Lookup ---
echo "[$(date)] === A: Hard Lookup ==="
python3 scripts/train.py $COMMON --hard_lookup
DIR_A=$(ls -dt logs/comet_solar_K16_conv1d_*hardlk*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_A" ] && [ ! -f "$DIR_A/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$DIR_A" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- B: No Revival ---
echo "[$(date)] === B: No Revival ==="
python3 scripts/train.py $COMMON --no_revival
DIR_B=$(ls -dt logs/comet_solar_K16_conv1d_*norev*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_B" ] && [ ! -f "$DIR_B/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$DIR_B" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- C: No EMA (freeze C after K-Means init) ---
echo "[$(date)] === C: No EMA ==="
python3 scripts/train.py $COMMON --no_ema
DIR_C=$(ls -dt logs/comet_solar_K16_conv1d_*noema*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_C" ] && [ ! -f "$DIR_C/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$DIR_C" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- E: No Entropy Reg ---
echo "[$(date)] === E: No Entropy Reg ==="
python3 scripts/train.py $COMMON --entropy_reg_weight 0
DIR_E=$(ls -dt logs/comet_solar_K16_conv1d_*noent*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_E" ] && [ ! -f "$DIR_E/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$DIR_E" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- F: No Revival + No Entropy ---
echo "[$(date)] === F: No Revival + No Entropy ==="
python3 scripts/train.py $COMMON --no_revival --entropy_reg_weight 0
DIR_F=$(ls -dt logs/comet_solar_K16_conv1d_*norev*noent*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_F" ] && [ ! -f "$DIR_F/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$DIR_F" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- G: Soft + Direct Add ---
echo "[$(date)] === G: Soft + Direct Add ==="
python3 scripts/train.py $COMMON --direct_add
DIR_G=$(ls -dt logs/comet_solar_K16_conv1d_*dadd*s0_* 2>/dev/null | grep -v hardlk | head -1)
if [ -n "$DIR_G" ] && [ ! -f "$DIR_G/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$DIR_G" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- H: Hard + Direct Add ---
echo "[$(date)] === H: Hard + Direct Add ==="
python3 scripts/train.py $COMMON --direct_add --hard_lookup
DIR_H=$(ls -dt logs/comet_solar_K16_conv1d_*hardlk*dadd*s0_* logs/comet_solar_K16_conv1d_*dadd*hardlk*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_H" ] && [ ! -f "$DIR_H/eval_100samples_mr0.85.json" ]; then
  python3 scripts/evaluate.py "$DIR_H" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- 결과 정리 ---
echo ""
echo "============================================================"
echo "[$(date)] Results Summary"
echo "============================================================"
python3 -c "
import json, glob, numpy as np

experiments = {
    'Baseline': 'logs/comet_solar_K16_conv1d_s0_*',
    'A: Hard Lookup': 'logs/comet_solar_K16_conv1d_*hardlk*s0_*',
    'B: No Revival': 'logs/comet_solar_K16_conv1d_*norev*s0_*',
    'C: No EMA': 'logs/comet_solar_K16_conv1d_*noema*s0_*',
    'E: No Entropy': 'logs/comet_solar_K16_conv1d_*noent*s0_*',
    'F: No Rev+Ent': 'logs/comet_solar_K16_conv1d_*norev*noent*s0_*',
    'G: Soft+DAdd': 'logs/comet_solar_K16_conv1d_*dadd*s0_*',
    'H: Hard+DAdd': 'logs/comet_solar_K16_conv1d_*hardlk*dadd*s0_*',
}

# Filter baseline: exclude ablation dirs
import os

print(f'{\"Experiment\":<20} {\"ObsMAE\":>10} {\"ObsRMSE\":>10} {\"ppl\":>6}')
print('-' * 50)

for name, pattern in experiments.items():
    dirs = sorted(glob.glob(pattern + '/eval_100samples_mr0.85.json'))
    if name == 'Baseline':
        dirs = [d for d in dirs if 'hardlk' not in d and 'norev' not in d and 'noema' not in d and 'noent' not in d and 'nocb' not in d]
    if dirs:
        d = json.load(open(dirs[-1]))
        # Get ppl from train_log
        log_dir = os.path.dirname(dirs[-1])
        try:
            tlog = json.load(open(os.path.join(log_dir, 'train_log.json')))
            epochs = tlog if isinstance(tlog, list) else tlog.get('epochs', tlog)
            ppl = epochs[-1].get('perplexity', 'N/A')
            if isinstance(ppl, float):
                ppl = f'{ppl:.1f}'
        except:
            ppl = 'N/A'
        print(f'{name:<20} {d[\"ObsMAE_mean\"]:>10.4f} {d[\"ObsRMSE_mean\"]:>10.4f} {ppl:>6}')
    else:
        print(f'{name:<20} {\"(pending)\":>10}')
"

echo ""
echo "[$(date)] All done!"
