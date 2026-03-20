#!/usr/bin/env python3
"""
Run GinAR baseline on COMET datasets with ObsMAE evaluation.
- Training: random mask per batch (matching COMET protocol)
- Eval: 100-mask ObsMAE (per-horizon average, matching Chauhan/VIDA)

Usage:
  python run_ginar.py --dataset solar --seed 0
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

GINAR_ROOT = None
for p in [Path("/home/elicer/VSF_Unified/external/ginar")]:
    if p.exists():
        GINAR_ROOT = p
        break
assert GINAR_ROOT is not None, "GinAR not found"
sys.path.insert(0, str(GINAR_ROOT))

from model1.ginar_arch import GinAR
from adjacent_matrix_norm import calculate_transition_matrix


def load_adj_from_pkl(file_path, adj_type="identity"):
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
    else:
        eye = np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)
        adj = [eye, eye]
    return adj


def inverse_norm(x, max_val, min_val):
    return x * (max_val - min_val) + min_val


def apply_random_mask(x, missing_rate, rng=None):
    """Zero out missing_rate fraction of variables. x: [B, T, N, C]"""
    B, T, N, C = x.shape
    x_masked = x.clone()
    n_mask = int(N * missing_rate)
    if rng is None:
        rng = np.random
    for b in range(B):
        mask_idx = rng.choice(N, size=n_mask, replace=False)
        x_masked[b, :, mask_idx, :] = 0.0
    return x_masked


def obs_mae_per_horizon(pred, true, obs_mask, null_val=0.0):
    """Per-horizon ObsMAE (Chauhan/VIDA compatible)."""
    B, Q, N = pred.shape
    pred_obs = pred[:, :, obs_mask]
    true_obs = true[:, :, obs_mask]
    h_maes, h_rmses = [], []
    for h in range(Q):
        p, t = pred_obs[:, h, :], true_obs[:, h, :]
        valid = (t != null_val)
        count = max(valid.sum().item(), 1)
        h_maes.append((torch.abs(p - t) * valid).sum().item() / count)
        h_rmses.append(math.sqrt(((p - t)**2 * valid).sum().item() / count))
    return float(np.mean(h_maes)), float(np.mean(h_rmses))


DATASET_CONFIGS = {
    "solar":   {"N": 137, "adj_type": "identity", "emb_size": 16, "grap_size": 8, "batch_size": 16, "null_val": 0.0},
    "ecg5000": {"N": 140, "adj_type": "identity", "emb_size": 16, "grap_size": 8, "batch_size": 16, "null_val": 0.0},
    "metr-la": {"N": 207, "adj_type": "doubletransition", "emb_size": 16, "grap_size": 8, "batch_size": 16, "null_val": 0.0},
    "traffic": {"N": 862, "adj_type": "identity", "emb_size": 16, "grap_size": 8, "batch_size": 4, "null_val": 0.0},
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
    null_val = cfg["null_val"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} (N={N})")

    # Load data
    data_path = Path(args.data_dir) / args.dataset / "data.npz"
    adj_path = Path(args.data_dir) / args.dataset / f"adj_{args.dataset}.pkl"
    assert data_path.exists(), f"Data not found: {data_path}"

    raw = np.load(data_path, allow_pickle=True)
    max_val, min_val = raw["max_min"]
    print(f"  Scale: max={max_val:.4f}, min={min_val:.4f}")

    # Load raw (unmasked) data — masking done per batch during training
    train_x_raw = torch.tensor(raw["train_x_raw"]).float()   # [n, N, T]
    train_x_raw = train_x_raw.transpose(1, 2).unsqueeze(-1)  # → [n, T, N, 1]
    train_y = torch.tensor(raw["train_y"]).float()            # [n, T, N, 1]
    train_data = torch.cat([train_x_raw, train_y], dim=-1)    # [n, T, N, 2]

    val_x_raw = torch.tensor(raw["vail_x_raw"]).float()      # [n, N, T]
    val_x_raw = val_x_raw.transpose(1, 2).unsqueeze(-1)      # → [n, T, N, 1]
    val_y = torch.tensor(raw["vail_y"]).float()
    val_data = torch.cat([val_x_raw, val_y], dim=-1)

    # Test data (raw, unmasked)
    test_x_raw = torch.tensor(raw["test_x_raw"]).float()     # [n, N, T]
    test_x_raw = test_x_raw.transpose(1, 2).unsqueeze(-1)    # → [n, T, N, 1]
    test_y = torch.tensor(raw["test_y"]).float()              # [n, T, N, 1]

    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False)

    in_size = 1
    adj = load_adj_from_pkl(str(adj_path), cfg["adj_type"])
    adj_mx = [torch.tensor(a).float() for a in adj]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_x_raw)}")

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
    milestone = [1, 15, 40, 70, 90]
    gamma = 0.5

    from metric.mask_metric import masked_mae

    # Train with random masking per batch
    log_dir = Path(args.log_dir) / f"ginar_{args.dataset}_s{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Training GinAR (seed={args.seed}, random mask per batch)")
    print(f"{'='*50}")

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 15
    patience_counter = 0
    train_rng = np.random.RandomState(args.seed + 1000)

    for epoch in range(args.epochs):
        model.train()
        train_loss, n_batch = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Apply random mask to input
            x_raw = batch[:, :, :, :in_size]  # [B, T, N, 1] unmasked
            x_masked = apply_random_mask(x_raw, args.missing_rate, train_rng)
            y = batch[:, :, :, -1].to(device)  # [B, T, N] future target

            pred = model(x_masked.to(device))
            loss = masked_mae(pred, y, null_val)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        train_loss /= n_batch

        # Validation (random mask, same as training)
        model.eval()
        val_loss, n_batch = 0.0, 0
        val_rng = np.random.RandomState(args.seed + epoch)
        with torch.no_grad():
            for batch in val_loader:
                x_raw = batch[:, :, :, :in_size]
                x_masked = apply_random_mask(x_raw, args.missing_rate, val_rng)
                y = batch[:, :, :, -1].to(device)
                pred = model(x_masked.to(device))
                loss = masked_mae(pred, y, null_val)
                val_loss += loss.item()
                n_batch += 1
        val_loss /= n_batch

        if (epoch + 1) in milestone:
            for pg in optimizer.param_groups:
                pg['lr'] *= gamma

        if (epoch + 1) % 10 == 0 or epoch == 0:
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
    # For each mask: mask test input → run model → compute ObsMAE
    print(f"\n{'='*50}")
    print(f"Evaluating ({args.n_eval_masks}-mask)")
    print(f"{'='*50}")

    test_dataset = torch.utils.data.TensorDataset(test_x_raw, test_y)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    results_mae, results_rmse = [], []
    t0 = time.time()

    for mask_i in range(args.n_eval_masks):
        mask_rng = np.random.RandomState(mask_i)
        n_obs = math.ceil(N * (1.0 - args.missing_rate))
        obs_idx = mask_rng.choice(N, size=n_obs, replace=False)
        obs_mask = torch.zeros(N, dtype=torch.bool)
        obs_mask[obs_idx] = True

        all_pred, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                # Apply this mask to input
                x_masked = x_batch.clone()
                x_masked[:, :, ~obs_mask, :] = 0.0
                pred = model(x_masked.to(device)).cpu()        # [B, 12, N]
                target = y_batch[:, :, :, 0]                    # [B, 12, N]
                all_pred.append(pred)
                all_true.append(target)

        all_pred = inverse_norm(torch.cat(all_pred, 0), max_val, min_val)
        all_true = inverse_norm(torch.cat(all_true, 0), max_val, min_val)

        mae, rmse = obs_mae_per_horizon(all_pred, all_true, obs_mask, null_val=null_val)
        results_mae.append(mae)
        results_rmse.append(rmse)

        if (mask_i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (mask_i + 1) * (args.n_eval_masks - mask_i - 1)
            print(f"  [{mask_i+1}/{args.n_eval_masks}] ObsMAE={mae:.4f} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    print(f"\nResults ({args.n_eval_masks} masks):")
    print(f"  ObsMAE:  {np.mean(results_mae):.4f} +/- {np.std(results_mae):.4f}")
    print(f"  ObsRMSE: {np.mean(results_rmse):.4f} +/- {np.std(results_rmse):.4f}")

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
        "all_ObsMAE": [float(x) for x in results_mae],
        "all_ObsRMSE": [float(x) for x in results_rmse],
        "n_params": n_params,
    }
    with open(log_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {log_dir / 'results.json'}")


if __name__ == "__main__":
    main()
