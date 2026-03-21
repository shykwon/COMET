#!/bin/bash
# Codebook Ablation v2: Solar seed 0, MTGNN, 100-mask eval
# All with EMA ON (disable_ema=false in default.yaml)
# 1. Baseline (soft + cross-attn + EMA)
# 2. Hard lookup + EMA (≈ VQ-VAE style)
# 3. No Entropy + EMA
# 4. Soft + Direct Add
# 5. Hard + Direct Add
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
echo "[$(date)] Codebook Ablation v2 (Solar, seed 0, EMA ON)"
echo "============================================================"

# --- 1. Baseline (soft + cross-attn + EMA) ---
echo "[$(date)] === 1. Baseline (soft + cross-attn + EMA) ==="
python3 scripts/train.py $COMMON
DIR_1=$(ls -dt logs/comet_solar_K16_conv1d_s0_* 2>/dev/null | grep -v nocb | grep -v hardlk | grep -v norev | grep -v noema | grep -v noent | grep -v dadd | grep -v film | head -1)
if [ -n "$DIR_1" ] && [ ! -f "$DIR_1/eval_100samples_mr0.85.json" ]; then
  echo "[$(date)] Eval $DIR_1"
  python3 scripts/evaluate.py "$DIR_1" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- 2. Hard Lookup + EMA (VQ-VAE style) ---
echo "[$(date)] === 2. Hard Lookup + EMA ==="
python3 scripts/train.py $COMMON --hard_lookup
DIR_2=$(ls -dt logs/comet_solar_K16_conv1d_*hardlk*s0_* 2>/dev/null | grep -v dadd | head -1)
if [ -n "$DIR_2" ] && [ ! -f "$DIR_2/eval_100samples_mr0.85.json" ]; then
  echo "[$(date)] Eval $DIR_2"
  python3 scripts/evaluate.py "$DIR_2" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- 3. No Entropy + EMA ---
echo "[$(date)] === 3. No Entropy + EMA ==="
python3 scripts/train.py $COMMON --entropy_reg_weight 0
DIR_3=$(ls -dt logs/comet_solar_K16_conv1d_*noent*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_3" ] && [ ! -f "$DIR_3/eval_100samples_mr0.85.json" ]; then
  echo "[$(date)] Eval $DIR_3"
  python3 scripts/evaluate.py "$DIR_3" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- 4. Soft + Direct Add ---
echo "[$(date)] === 4. Soft + Direct Add ==="
python3 scripts/train.py $COMMON --direct_add
DIR_4=$(ls -dt logs/comet_solar_K16_conv1d_*dadd*s0_* 2>/dev/null | grep -v hardlk | head -1)
if [ -n "$DIR_4" ] && [ ! -f "$DIR_4/eval_100samples_mr0.85.json" ]; then
  echo "[$(date)] Eval $DIR_4"
  python3 scripts/evaluate.py "$DIR_4" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- 5. Hard + Direct Add ---
echo "[$(date)] === 5. Hard + Direct Add ==="
python3 scripts/train.py $COMMON --direct_add --hard_lookup
DIR_5=$(ls -dt logs/comet_solar_K16_conv1d_*hardlk*dadd*s0_* logs/comet_solar_K16_conv1d_*dadd*hardlk*s0_* 2>/dev/null | head -1)
if [ -n "$DIR_5" ] && [ ! -f "$DIR_5/eval_100samples_mr0.85.json" ]; then
  echo "[$(date)] Eval $DIR_5"
  python3 scripts/evaluate.py "$DIR_5" --missing_rate 0.85 --n_samples 100 --batch_size 64 --amp_bf16
fi

# --- 결과 정리 ---
echo ""
echo "============================================================"
echo "[$(date)] Results Summary"
echo "============================================================"
python3 -c "
import json, glob, os, numpy as np

experiments = {
    '1. Baseline (EMA)':     ('logs/comet_solar_K16_conv1d_s0_*', ['nocb','hardlk','norev','noema','noent','dadd','film']),
    '2. Hard+EMA (VQ)':      ('logs/comet_solar_K16_conv1d_*hardlk*s0_*', ['dadd']),
    '3. No Entropy+EMA':     ('logs/comet_solar_K16_conv1d_*noent*s0_*', ['norev']),
    '4. Soft+DAdd':          ('logs/comet_solar_K16_conv1d_*dadd*s0_*', ['hardlk']),
    '5. Hard+DAdd':          ('logs/comet_solar_K16_conv1d_*hardlk*dadd*s0_*', []),
}

print(f'{\"Experiment\":<25} {\"ObsMAE\":>10} {\"ObsRMSE\":>10} {\"ppl\":>6}')
print('-' * 55)

for name, (pattern, excludes) in experiments.items():
    files = sorted(glob.glob(pattern + '/eval_100samples_mr0.85.json'))
    for ex in excludes:
        files = [f for f in files if ex not in f]
    if files:
        d = json.load(open(files[-1]))
        log_dir = os.path.dirname(files[-1])
        try:
            tlog = json.load(open(os.path.join(log_dir, 'train_log.json')))
            epochs = tlog if isinstance(tlog, list) else tlog.get('epochs', tlog)
            ppl = epochs[-1].get('perplexity', 'N/A')
            if isinstance(ppl, float): ppl = f'{ppl:.1f}'
        except: ppl = 'N/A'
        print(f'{name:<25} {d[\"ObsMAE_mean\"]:>10.4f} {d[\"ObsRMSE_mean\"]:>10.4f} {ppl:>6}')
    else:
        print(f'{name:<25} {\"(pending)\":>10}')
"

echo ""
echo "[$(date)] All done!"
