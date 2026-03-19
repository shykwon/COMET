#!/usr/bin/env python3
"""
Run GinAR baseline on COMET datasets with ObsMAE evaluation.

Usage:
  python run_ginar.py --dataset solar --seed 0
  python run_ginar.py --dataset ecg5000 --seed 0 --n_eval_masks 100
"""

import os
import sys
import argparse
import math
import json
import time
import numpy as np
import pickle
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add GinAR to path
GINAR_ROOT = Path(__file__).resolve().parent.parent.parent / "VSF_Unified" / "external" / "ginar"
# Try parent paths
for p in [GINAR_ROOT, Path("/home/elicer/VSF_Unified/external/ginar")]:
    if p.exists():
        GINAR_ROOT = p
        break
sys.path.insert(0, str(GINAR_ROOT))

from model1.ginar_arch import GinAR
from adjacent_matrix_norm import calculate_transition_matrix


def load_adj_from_pkl(file_path, adj_type="identity"):
    """Load adjacency matrix."""
    with open(file_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    if isinstance(obj, (list, tuple)):
        adj_mx = obj[-1] if len(obj) == 3 else obj[0]
    else:
        adj_mx = obj

    adj_mx = np.array(adj_mx, dtype=np.float32)

    if adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T,
               calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        # GinAR expects 2 adj matrices (forward + backward)
        eye = np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)
        adj = [eye, eye]
    else:
        adj = [adj_mx, adj_mx]
    return adj


def inverse_norm(x, max_val, min_val):
    return x * (max_val - min_val) + min_val


def obs_mae_per_horizon(pred, true, obs_mask, null_val=0.0):
    """Compute per-horizon ObsMAE (COMET/VIDA compatible)."""
    # pred, true: [B, Q, N]
    # obs_mask: [N] boolean
    B, Q, N = pred.shape
    pred_obs = pred[:, :, obs_mask]
    true_obs = true[:, :, obs_mask]

    h_maes, h_rmses = [], []
    for h in range(Q):
        p = pred_obs[:, h, :]
        t = true_obs[:, h, :]
        valid = (t != null_val)
        count = max(valid.sum().item(), 1)
        mae = (torch.abs(p - t) * valid).sum().item() / count
        rmse = math.sqrt(((p - t)**2 * valid).sum().item() / count)
        h_maes.append(mae)
        h_rmses.append(rmse)

    return float(np.mean(h_maes)), float(np.mean(h_rmses))


DATASET_CONFIGS = {
    "solar":   {"N": 137, "adj_type": "identity", "emb_size": 16, "grap_size": 8, "batch_size": 16},
    "ecg5000": {"N": 140, "adj_type": "identity", "emb_size": 16, "grap_size": 8, "batch_size": 16},
    "metr-la": {"N": 207, "adj_type": "doubletransition", "emb_size": 16, "grap_size": 8, "batch_size": 16},
    "traffic": {"N": 862, "adj_type": "identity", "emb_size": 16, "grap_size": 8, "batch_size": 4},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--missing_rate", type=float, default=0.85)
    parser.add_argument("--n_eval_masks", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--log_dir", type=str, default=str(Path(__file__).parent / "logs"))
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    N = cfg["N"]

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} (N={N})")

    # Load data
    data_path = Path(args.data_dir) / args.dataset / "data.npz"
    adj_path = Path(args.data_dir) / args.dataset / f"adj_{args.dataset}.pkl"

    assert data_path.exists(), f"Data not found: {data_path}. Run prepare_data.py first."

    raw = np.load(data_path, allow_pickle=True)
    max_val, min_val = raw["max_min"]

    # Load data
    # GinAR format: x_mask [n, T, N, 1], y [n, T, N, 1]
    # concat along dim=-1 → [n, T, N, 2]
    # x = batch[:, :, :, 0:1] (masked input), y = batch[:, :, :, -1] (target)
    mask_key = f"train_x_mask_{int(args.missing_rate * 100)}"
    val_mask_key = f"vail_x_mask_{int(args.missing_rate * 100)}"

    train_x = torch.tensor(raw.get(mask_key, raw["train_x_raw"])).float()  # use raw if mask not found
    train_data = torch.cat([train_x, torch.tensor(raw["train_y"]).float()], dim=-1)

    val_x = torch.tensor(raw.get(val_mask_key, raw["vail_x_raw"])).float()
    val_data = torch.cat([val_x, torch.tensor(raw["vail_y"]).float()], dim=-1)

    # For test: use raw (unmasked), masking done at eval time per ObsMAE protocol
    test_x = torch.tensor(raw["test_x_raw"]).float()  # [n, N, T] → need to reshape
    test_y = torch.tensor(raw["test_y"]).float()       # [n, T, N, 1]

    # test_x is [n, N, T] from feature_target (not transposed+expanded yet)
    # Need to match format: [n, T, N, 1]
    test_x_formatted = test_x.transpose(1, 2).unsqueeze(-1)  # [n, N, T] → [n, T, N, 1]
    test_data_raw = torch.cat([test_x_formatted, test_y], dim=-1)  # [n, T, N, 2]

    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data_raw, batch_size=cfg["batch_size"], shuffle=False)

    in_size = 1  # value only (feature dim)

    # Adj
    adj = load_adj_from_pkl(str(adj_path), cfg["adj_type"])
    adj_mx = [torch.tensor(a).float() for a in adj]

    print(f"  Train samples: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data_raw)}")

    # Model
    model = GinAR(
        input_len=12, num_id=N, out_len=12,
        in_size=in_size, emb_size=cfg["emb_size"],
        grap_size=cfg["grap_size"], layer_num=2,
        dropout=0.15, adj_mx=adj_mx
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GinAR params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # LR schedule (from GinAR original)
    milestone = [1, 15, 40, 70, 90]
    gamma = 0.5

    # Train
    print(f"\n{'='*50}")
    print(f"Training GinAR (seed={args.seed})")
    print(f"{'='*50}")

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 15
    patience_counter = 0

    log_dir = Path(args.log_dir) / f"ginar_{args.dataset}_s{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batch = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[:, :, :, :in_size].to(device)   # [B, T=12, N, 1] (masked input)
            y = batch[:, :, :, -1].to(device)           # [B, T=12, N] (target = future)
            pred = model(x)                              # [B, out_len=12, N]
            from metric.mask_metric import masked_mae
            loss = masked_mae(pred, y, 0.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        train_loss /= n_batch

        # Validation
        model.eval()
        val_loss = 0.0
        n_batch = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[:, :, :, :in_size].to(device)
                y = batch[:, :, :, -1].to(device)
                pred = model(x)
                loss = masked_mae(pred, y, 0.0)
                val_loss += loss.item()
                n_batch += 1
        val_loss /= n_batch

        # LR schedule
        if (epoch + 1) in milestone:
            for pg in optimizer.param_groups:
                pg['lr'] *= gamma

        print(f"  Epoch {epoch+1}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), log_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best={best_epoch})")
                break

    # Load best model
    model.load_state_dict(torch.load(log_dir / "best_model.pt", map_location=device))
    model.eval()

    # 100-mask ObsMAE evaluation
    print(f"\n{'='*50}")
    print(f"Evaluating ({args.n_eval_masks}-mask)")
    print(f"{'='*50}")

    # Collect all test predictions (using raw unmasked input for test)
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[:, :, :, :in_size].to(device)  # [B, T, N, 1]
            y = batch[:, :, :, -1]                     # [B, T, N] target
            pred = model(x).cpu()                      # [B, out_len, N]
            all_pred.append(pred)
            all_true.append(y)
    all_pred = torch.cat(all_pred, dim=0)  # [n_test, 12, N]
    all_true = torch.cat(all_true, dim=0)

    # Inverse normalize
    all_pred_inv = inverse_norm(all_pred, max_val, min_val)
    all_true_inv = inverse_norm(all_true, max_val, min_val)

    results_mae, results_rmse = [], []
    t0 = time.time()

    for i in range(args.n_eval_masks):
        rng = np.random.RandomState(i)
        n_obs = math.ceil(N * (1.0 - args.missing_rate))
        obs_idx = rng.choice(N, size=n_obs, replace=False)
        obs_mask = torch.zeros(N, dtype=torch.bool)
        obs_mask[obs_idx] = True

        mae, rmse = obs_mae_per_horizon(all_pred_inv, all_true_inv, obs_mask, null_val=0.0)
        results_mae.append(mae)
        results_rmse.append(rmse)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{args.n_eval_masks}] ObsMAE={mae:.4f} ObsRMSE={rmse:.4f} ({elapsed:.0f}s)")

    print(f"\nResults ({args.n_eval_masks} masks):")
    print(f"  ObsMAE:  {np.mean(results_mae):.4f} +/- {np.std(results_mae):.4f}")
    print(f"  ObsRMSE: {np.mean(results_rmse):.4f} +/- {np.std(results_rmse):.4f}")

    # Save results
    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "missing_rate": args.missing_rate,
        "n_eval_masks": args.n_eval_masks,
        "best_epoch": best_epoch,
        "ObsMAE_mean": float(np.mean(results_mae)),
        "ObsMAE_std": float(np.std(results_mae)),
        "ObsRMSE_mean": float(np.mean(results_rmse)),
        "ObsRMSE_std": float(np.std(results_rmse)),
        "all_ObsMAE": results_mae,
        "all_ObsRMSE": results_rmse,
        "n_params": n_params,
    }
    with open(log_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {log_dir / 'results.json'}")


if __name__ == "__main__":
    main()
