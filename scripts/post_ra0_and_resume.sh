#!/bin/bash
# Wait for ra0 eval to finish, compile all ETTh1 results, then resume ETTm1
export PYTHONUNBUFFERED=1

# ============================================================
# Step 1: Wait for ra0 experiment to finish
# ============================================================
RA0_PID=$(pgrep -f "run_etth1_no_ra")
if [ -n "$RA0_PID" ]; then
  echo "[$(date)] Waiting for ra0 experiment (PID: $RA0_PID) to finish..."
  while kill -0 "$RA0_PID" 2>/dev/null; do
    sleep 30
  done
fi
echo "[$(date)] ra0 experiment finished."

# ============================================================
# Step 2: Verify ra0 results (10 seeds)
# ============================================================
echo ""
echo "============================================================"
echo "Verifying ra0 experiment completeness"
echo "============================================================"

RA0_COUNT=$(ls logs/comet_ETTh1_K8_conv1d_ra0_*/eval_100samples_mr0.85.json 2>/dev/null | wc -l)
echo "ra0 evals completed: $RA0_COUNT / 10"

if [ "$RA0_COUNT" -lt 10 ]; then
  echo "[WARN] Not all ra0 evals completed. Running remaining..."
  bash scripts/eval_all.sh --dataset ETTh1 --pattern "ra0" --cpu
fi

# ============================================================
# Step 3: Compile ALL ETTh1 results (original + ra0)
# ============================================================
echo ""
echo "============================================================"
echo "Compiling ETTh1 full results"
echo "============================================================"

python3 << 'PYEOF'
import json, os, yaml
from pathlib import Path
from collections import defaultdict
import numpy as np

results = defaultdict(lambda: defaultdict(list))

for d in sorted(Path("logs").glob("comet_ETTh1_*")):
    eval_file = d / "eval_100samples_mr0.85.json"
    config_file = d / "config.yaml"
    if not eval_file.exists() or not config_file.exists():
        print(f"  [SKIP] {d.name} (missing eval or config)")
        continue

    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    with open(eval_file) as f:
        ev = json.load(f)

    head = cfg["model"].get("head_type", "mtgnn")
    use_cb = cfg["model"].get("use_codebook", True)
    ra = cfg["model"].get("restore_alpha", 0.1)

    if not use_cb:
        condition = "nocb"
    elif ra == 0:
        condition = "comet_ra0"
    else:
        condition = "comet"
    key = f"{head}_{condition}"

    results[key]["ObsMAE"].append(ev["ObsMAE_mean"])
    results[key]["ObsRMSE"].append(ev["ObsRMSE_mean"])

# Print results table
print(f"\n{'='*70}")
print(f"ETTh1 Results Summary (100-mask, 10-seed)")
print(f"{'='*70}")
print(f"{'Method':<25} {'N':>3} {'ObsMAE':>15} {'ObsRMSE':>15}")
print(f"{'-'*70}")

order = ["mtgnn_comet", "mtgnn_comet_ra0", "mtgnn_nocb",
         "astgcn_comet", "astgcn_nocb",
         "mstgcn_comet", "mstgcn_nocb",
         "tgcn_comet", "tgcn_nocb"]

for key in order:
    if key not in results:
        continue
    maes = results[key]["ObsMAE"]
    rmses = results[key]["ObsRMSE"]
    n = len(maes)
    print(f"{key:<25} {n:>3} {np.mean(maes):>7.4f}±{np.std(maes):.4f} {np.mean(rmses):>7.4f}±{np.std(rmses):.4f}")

# Save to JSON
output = {
    "description": "RQ1 ETTh1 results (incl. restore_alpha=0 ablation)",
    "dataset": "ETTh1",
    "protocol": "10-seed x 100-mask",
    "runs": {}
}
for key in sorted(results.keys()):
    maes = results[key]["ObsMAE"]
    rmses = results[key]["ObsRMSE"]
    output["runs"][key] = {
        "n_seeds": len(maes),
        "test": {
            "ObsMAE": {"mean": round(float(np.mean(maes)), 4), "std": round(float(np.std(maes)), 4)},
            "ObsRMSE": {"mean": round(float(np.mean(rmses)), 4), "std": round(float(np.std(rmses)), 4)},
        }
    }

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/RQ1_ETTh1.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n{'='*70}")
print("Saved to experiments/results/RQ1_ETTh1.json")
print(f"{'='*70}")

# Double-check: verify seed counts
print(f"\n--- Double Check ---")
for key in order:
    if key not in results:
        continue
    n = len(results[key]["ObsMAE"])
    status = "OK" if n == 10 else f"WARN: only {n} seeds!"
    print(f"  {key:<25} seeds={n}  [{status}]")
PYEOF

# ============================================================
# Step 4: Resume ETTm1 (from seed 4 onwards, skip completed)
# ============================================================
echo ""
echo "============================================================"
echo "Resuming ETTm1 full pipeline"
echo "============================================================"

# Modify ETTm1 script to resume from seed 4 (seeds 0-3 already done for mtgnn)
# Use a custom resume script instead
nohup bash scripts/run_ettm1_resume.sh > logs/ettm1_main_run.log 2>&1 &
ETTM1_PID=$!
echo "ETTm1 started (PID: $ETTM1_PID)"
echo "[$(date)] Done."
