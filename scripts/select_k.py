#!/usr/bin/env python3
"""K selection via Elbow method on Stage 1 Q_full embeddings.

Trains Stage 1 only (10 epochs) to collect Q_full embeddings,
then runs K-Means for multiple K values and plots the elbow curve.

Usage:
    python scripts/select_k.py --dataset solar --data_dir ./data
    python scripts/select_k.py --dataset solar --data_dir ./data --k_candidates 4 8 16 32 64 128
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comet.models.comet import COMET
from comet.data.dataset import create_dataloaders
from comet.training.curriculum import CurriculumScheduler, apply_masking
from comet.training.losses import compute_infonce


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_kmeans(embeddings: torch.Tensor, K: int, n_iter: int = 50):
    """Run K-Means and return centroids + inertia (SSE)."""
    N = embeddings.shape[0]
    indices = torch.randperm(N, device=embeddings.device)[:K]
    centroids = embeddings[indices].clone()

    for _ in range(n_iter):
        dists = torch.cdist(embeddings, centroids)  # [N, K]
        assignments = dists.argmin(dim=-1)           # [N]
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                centroids[k] = embeddings[mask].mean(dim=0)

    # Compute inertia (within-cluster sum of squared distances)
    dists = torch.cdist(embeddings, centroids)
    min_dists = dists.min(dim=-1).values
    inertia = (min_dists ** 2).sum().item()

    return centroids, inertia, assignments


def compute_silhouette_approx(embeddings: torch.Tensor, assignments: torch.Tensor, K: int,
                               max_samples: int = 5000):
    """Approximate silhouette score (subsample for speed)."""
    N = embeddings.shape[0]
    if N > max_samples:
        idx = torch.randperm(N, device=embeddings.device)[:max_samples]
        embeddings = embeddings[idx]
        assignments = assignments[idx]
        N = max_samples

    # Compute pairwise distances
    dists = torch.cdist(embeddings, embeddings)  # [N, N]

    silhouettes = []
    for k in range(K):
        mask_k = assignments == k
        n_k = mask_k.sum().item()
        if n_k <= 1:
            continue

        # a(i): mean intra-cluster distance
        intra = dists[mask_k][:, mask_k]
        a = intra.sum(dim=-1) / (n_k - 1)

        # b(i): min mean inter-cluster distance
        b = torch.full((n_k,), float('inf'), device=embeddings.device)
        for j in range(K):
            if j == k:
                continue
            mask_j = assignments == j
            if mask_j.sum() == 0:
                continue
            inter = dists[mask_k][:, mask_j].mean(dim=-1)
            b = torch.min(b, inter)

        s = (b - a) / torch.max(a, b).clamp(min=1e-8)
        silhouettes.append(s.mean().item())

    return np.mean(silhouettes) if silhouettes else 0.0


def main():
    parser = argparse.ArgumentParser(description="Select optimal K via Elbow method")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--k_candidates", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--n_kmeans_runs", type=int, default=5,
                        help="Number of K-Means runs per K (best inertia selected)")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "experiments/results"))
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["dataset"] = args.dataset
    cfg["data"]["data_dir"] = args.data_dir
    cfg["training"]["seed"] = args.seed
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"K candidates: {args.k_candidates}")

    # Data
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    train_loader, val_loader, _, num_variates, scaler = create_dataloaders(
        dataset_name=data_cfg["dataset"],
        data_dir=data_cfg["data_dir"],
        seq_len=data_cfg["seq_len"],
        pred_len=data_cfg["pred_len"],
        batch_size=train_cfg["batch_size"],
        num_workers=cfg["hardware"].get("num_workers", 4),
        pin_memory=cfg["hardware"].get("pin_memory", True),
        global_scaler=data_cfg.get("global_scaler", False),
    )

    # Model (use a temporary K, doesn't matter for Stage 1)
    cb_cfg = model_cfg["codebook"]
    model = COMET(
        num_variates=num_variates,
        seq_len=data_cfg["seq_len"],
        pred_len=data_cfg["pred_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=model_cfg.get("n_encoder_layers", 2),
        codebook_K=cb_cfg["K"],
        codebook_tau=cb_cfg["tau"],
        codebook_ema_alpha=cb_cfg["ema_alpha"],
        patch_len=model_cfg["patch_len"],
        stride=model_cfg["stride"],
        dropout=model_cfg["dropout"],
        temporal_config=model_cfg.get("temporal"),
        use_codebook=model_cfg.get("use_codebook", True),
        head_type=model_cfg.get("head_type", "mtgnn"),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: COMET ({n_params:,} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                   weight_decay=train_cfg["weight_decay"])

    stage1_epochs = train_cfg.get("stage1_epochs", 10)
    loss_fn = train_cfg.get("loss_fn", "mae")
    null_val = train_cfg.get("null_val")
    denorm_loss = train_cfg.get("denorm_loss", False)

    if scaler is not None:
        sc_std = torch.tensor(scaler.std.squeeze(), dtype=torch.float32, device=device)
        sc_mean = torch.tensor(scaler.mean.squeeze(), dtype=torch.float32, device=device)
    else:
        sc_std = sc_mean = None

    # -----------------------------------------------------------------------
    # Stage 1: Collect Q_full embeddings
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Stage 1: Training {stage1_epochs} epochs to collect Q_full embeddings")
    print(f"{'='*60}")

    q_buffer = []
    daytime_buffer = []

    for epoch in range(1, stage1_epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            y_raw = batch[2].to(device) if len(batch) > 2 else None
            B, N, T = x.shape

            # Stage 1: no masking
            obs_mask = torch.ones(B, N, dtype=torch.bool, device=device)

            y_hat, Q_sub, w_sub = model(x, obs_mask)

            with torch.no_grad():
                _, Q_full, _ = model.forward_full(x)
                q_buffer.append(Q_full.detach().cpu())

                # Daytime detection
                if sc_std is not None:
                    y_orig = y * sc_std.unsqueeze(0).unsqueeze(-1) + sc_mean.unsqueeze(0).unsqueeze(-1)
                    is_daytime = (y_orig.abs() > 0.1).float().mean(dim=[1, 2]) > 0.3
                    daytime_buffer.append(is_daytime.detach().cpu())

            # Task loss
            y_hat_l, y_l = y_hat, y
            if denorm_loss and sc_std is not None:
                std_e = sc_std.unsqueeze(0).unsqueeze(-1)
                mean_e = sc_mean.unsqueeze(0).unsqueeze(-1)
                y_hat_l = y_hat_l * std_e + mean_e
                y_l = y_l * std_e + mean_e

            if loss_fn == "mae":
                errors = torch.abs(y_hat_l - y_l)
            else:
                errors = (y_hat_l - y_l).pow(2)

            if null_val is not None:
                if y_raw is not None:
                    nv_mask = y_raw != null_val
                else:
                    nv_mask = y_l.abs() > 1e-4
                nv_f = nv_mask.float()
                loss = (errors * nv_f).sum() / nv_f.sum().clamp(min=1.0)
            else:
                loss = errors.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        dt = time.time() - t0
        print(f"  Epoch {epoch}/{stage1_epochs} | loss={epoch_loss/n_batches:.4f} | {dt:.0f}s")

    # -----------------------------------------------------------------------
    # K-Means Elbow Analysis
    # -----------------------------------------------------------------------
    Q_all = torch.cat(q_buffer).to(device)
    print(f"\nCollected {Q_all.shape[0]} Q_full embeddings, dim={Q_all.shape[1]}")

    # Filter daytime if available
    if daytime_buffer:
        is_day = torch.cat(daytime_buffer).to(device)
        n_day = is_day.sum().item()
        if n_day >= max(args.k_candidates):
            Q_all = Q_all[is_day]
            print(f"  Filtered to {Q_all.shape[0]} daytime embeddings")

    print(f"\n{'='*60}")
    print(f"Elbow Analysis: K = {args.k_candidates}")
    print(f"{'='*60}")

    results = []
    for K in args.k_candidates:
        best_inertia = float('inf')
        best_assignments = None

        for run in range(args.n_kmeans_runs):
            centroids, inertia, assignments = run_kmeans(Q_all, K)
            if inertia < best_inertia:
                best_inertia = inertia
                best_assignments = assignments

        silhouette = compute_silhouette_approx(Q_all, best_assignments, K)

        results.append({
            "K": K,
            "inertia": best_inertia,
            "silhouette": silhouette,
        })
        print(f"  K={K:3d} | Inertia={best_inertia:12.1f} | Silhouette={silhouette:.4f}")

    # -----------------------------------------------------------------------
    # Find elbow point (maximum second derivative)
    # -----------------------------------------------------------------------
    inertias = [r["inertia"] for r in results]
    Ks = [r["K"] for r in results]

    if len(Ks) >= 3:
        # Compute second differences
        d1 = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        d2 = [d1[i] - d1[i+1] for i in range(len(d1)-1)]
        elbow_idx = np.argmax(d2) + 1  # +1 because d2 starts from index 1
        elbow_K = Ks[elbow_idx]
    else:
        elbow_K = Ks[0]

    # Also check silhouette-based recommendation
    sil_scores = [r["silhouette"] for r in results]
    sil_best_K = Ks[np.argmax(sil_scores)]

    print(f"\n{'='*60}")
    print(f"Recommendation")
    print(f"{'='*60}")
    print(f"  Elbow method:    K = {elbow_K}")
    print(f"  Silhouette best: K = {sil_best_K}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": args.dataset,
        "seed": args.seed,
        "stage1_epochs": stage1_epochs,
        "n_embeddings": Q_all.shape[0],
        "d_model": Q_all.shape[1],
        "k_candidates": args.k_candidates,
        "n_kmeans_runs": args.n_kmeans_runs,
        "results": results,
        "recommendation": {
            "elbow_K": int(elbow_K),
            "silhouette_K": int(sil_best_K),
        },
    }

    out_path = output_dir / f"k_selection_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # -----------------------------------------------------------------------
    # Plot (optional, if matplotlib available)
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Elbow plot
        ax1.plot(Ks, inertias, "bo-", linewidth=2, markersize=8)
        ax1.axvline(x=elbow_K, color="r", linestyle="--", label=f"Elbow K={elbow_K}")
        ax1.set_xlabel("K (codebook size)")
        ax1.set_ylabel("Inertia (SSE)")
        ax1.set_title(f"Elbow Method — {args.dataset}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette plot
        ax2.plot(Ks, sil_scores, "go-", linewidth=2, markersize=8)
        ax2.axvline(x=sil_best_K, color="r", linestyle="--", label=f"Best K={sil_best_K}")
        ax2.set_xlabel("K (codebook size)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title(f"Silhouette Score — {args.dataset}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / f"k_selection_{args.dataset}.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved to {fig_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
