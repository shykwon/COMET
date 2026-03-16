#!/usr/bin/env python3
"""Compile experiment results in VIDA-compatible format (10-seed × 100-mask = 1000 runs)."""

import json
import glob
import sys
import numpy as np
from pathlib import Path


def compile_runs(pattern, label=""):
    """Compile results from multiple seed directories matching pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    all_mae, all_rmse = [], []
    seed_maes, seed_rmses = [], []

    for f in files:
        d = json.load(open(f))
        all_mae.extend(d['all_ObsMAE'])
        all_rmse.extend(d['all_ObsRMSE'])
        seed_maes.append(d['ObsMAE_mean'])
        seed_rmses.append(d['ObsRMSE_mean'])

    return {
        "label": label,
        "n_seeds": len(files),
        "n_total_runs": len(all_mae),
        # VIDA-compatible: 1000 runs 통합
        "ObsMAE": f"{np.mean(all_mae):.2f}({np.std(all_mae):.2f})",
        "ObsRMSE": f"{np.mean(all_rmse):.2f}({np.std(all_rmse):.2f})",
        "ObsMAE_mean": float(np.mean(all_mae)),
        "ObsMAE_std": float(np.std(all_mae)),
        "ObsRMSE_mean": float(np.mean(all_rmse)),
        "ObsRMSE_std": float(np.std(all_rmse)),
        # Seed-level stats (for reference)
        "seed_ObsMAE_mean": float(np.mean(seed_maes)),
        "seed_ObsMAE_std": float(np.std(seed_maes)),
        "seed_ObsRMSE_mean": float(np.mean(seed_rmses)),
        "seed_ObsRMSE_std": float(np.std(seed_rmses)),
    }


def print_table(results, dataset):
    """Print results in paper-ready table format."""
    print(f"\n{'='*70}")
    print(f"  {dataset} Results (VIDA-compatible: 10-seed × 100-mask)")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'ObsMAE':<20} {'ObsRMSE':<20} {'runs'}")
    print(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*5}")
    for r in results:
        if r is not None:
            print(f"  {r['label']:<25} {r['ObsMAE']:<20} {r['ObsRMSE']:<20} {r['n_total_runs']}")


if __name__ == "__main__":
    # ECG5000
    ecg_results = []
    for head in ['mtgnn', 'astgcn', 'mstgcn', 'tgcn']:
        if head == 'mtgnn':
            comet_pat = 'logs/comet_ecg5000_K16_conv1d_ra0_s*/eval_100samples_mr0.85.json'
            nocb_pat = 'logs/comet_ecg5000_K16_conv1d_nocb_s*/eval_100samples_mr0.85.json'
        else:
            comet_pat = f'logs/comet_ecg5000_K16_conv1d_{head}_s*/eval_100samples_mr0.85.json'
            nocb_pat = f'logs/comet_ecg5000_K16_conv1d_{head}_nocb_s*/eval_100samples_mr0.85.json'

        r_comet = compile_runs(comet_pat, f"{head.upper()} COMET")
        r_nocb = compile_runs(nocb_pat, f"{head.upper()} nocb")
        if r_comet:
            ecg_results.append(r_comet)
        if r_nocb:
            ecg_results.append(r_nocb)

    if ecg_results:
        print_table(ecg_results, "ECG5000")

    # ETTh1
    etth1_results = []
    r = compile_runs('logs/comet_ETTh1_K8_conv1d_ra0_s*/eval_100samples_mr0.85.json', 'MTGNN COMET')
    if r:
        etth1_results.append(r)
    r = compile_runs('logs/comet_ETTh1_K8_conv1d_nocb_s*/eval_100samples_mr0.85.json', 'MTGNN nocb')
    if r:
        etth1_results.append(r)

    if etth1_results:
        print_table(etth1_results, "ETTh1")

    # Solar
    solar_results = []
    r = compile_runs('logs/comet_solar_K16_conv1d_s*/eval_100samples_mr0.85.json', 'MTGNN COMET')
    if r:
        solar_results.append(r)
    r = compile_runs('logs/comet_solar_K16_conv1d_nocb_s*/eval_100samples_mr0.85.json', 'MTGNN nocb')
    if r:
        solar_results.append(r)

    if solar_results:
        print_table(solar_results, "Solar")

    # Save all to JSON
    all_results = {}
    if ecg_results:
        all_results['ecg5000'] = ecg_results
    if etth1_results:
        all_results['ETTh1'] = etth1_results
    if solar_results:
        all_results['solar'] = solar_results

    out_path = 'experiments/results/compiled_all.json'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
