#!/usr/bin/env python3
"""N-times random sensor mask evaluation for paper-ready results."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comet.models.comet import COMET
from comet.data.dataset import create_dataloaders


def evaluate_once(model, loader, device, scaler, missing_rate, seed, num_variates):
    """Single evaluation with a fixed sensor mask."""
    model.eval()
    preds_list, trues_list, raw_list = [], [], []

    rng = np.random.RandomState(seed)
    n_observed = math.ceil(num_variates * (1.0 - missing_rate))
    observed_idx = rng.choice(num_variates, size=n_observed, replace=False)
    fixed_mask = torch.zeros(num_variates, dtype=torch.bool, device=device)
    fixed_mask[observed_idx] = True

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            B = x.shape[0]
            obs_mask = fixed_mask.unsqueeze(0).expand(B, -1)
            y_hat, _, _, _ = model(x, obs_mask)
            preds_list.append(y_hat.cpu().numpy())
            trues_list.append(y.cpu().numpy())
            if len(batch) > 2:
                raw_list.append(batch[2].numpy())

    preds = np.concatenate(preds_list)
    trues = np.concatenate(trues_list)
    masks = fixed_mask.cpu().numpy()
    shape = preds.shape

    preds_flat = preds.transpose(0, 2, 1).reshape(-1, shape[1])
    trues_flat = trues.transpose(0, 2, 1).reshape(-1, shape[1])
    preds_inv = scaler.inverse_transform(preds_flat).reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)
    trues_inv = scaler.inverse_transform(trues_flat).reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)

    mask_exp = np.expand_dims(masks, (0, -1)).repeat(shape[0], axis=0).repeat(shape[2], axis=-1)

    if raw_list:
        nonnull = (np.concatenate(raw_list) != 0.0).astype(float)
    else:
        nonnull = (trues_inv != 0.0).astype(float)
    valid = mask_exp * nonnull

    abs_err = np.abs(preds_inv - trues_inv) * valid
    sq_err = (preds_inv - trues_inv) ** 2 * valid
    count = valid.sum()

    obs_mae = float(abs_err.sum() / count)
    obs_rmse = float(np.sqrt(sq_err.sum() / count))

    per_horizon = {}
    for t in range(shape[2]):
        h_count = valid[:, :, t].sum()
        if h_count > 0:
            per_horizon[t] = {
                "MAE": float(abs_err[:, :, t].sum() / h_count),
                "RMSE": float(np.sqrt(sq_err[:, :, t].sum() / h_count)),
            }

    return {"ObsMAE": obs_mae, "ObsRMSE": obs_rmse, "per_horizon": per_horizon}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("--missing_rate", type=float, default=0.85)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    model_path = exp_dir / "best_model.pt"

    config_path = exp_dir / "config.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"
    assert model_path.exists(), f"Model not found: {model_path}"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {exp_dir}")
    print(f"Missing rate: {args.missing_rate}, N samples: {args.n_samples}")

    data_cfg = cfg["data"]
    _, _, test_loader, num_variates, scaler = create_dataloaders(
        dataset_name=data_cfg["dataset"],
        data_dir=data_cfg["data_dir"],
        seq_len=data_cfg["seq_len"],
        pred_len=data_cfg["pred_len"],
        batch_size=args.batch_size,
        global_scaler=data_cfg.get("global_scaler", False),
        num_workers=4, pin_memory=True,
    )

    cb_cfg = cfg["model"]["codebook"]
    model = COMET(
        num_variates=num_variates,
        seq_len=data_cfg["seq_len"],
        pred_len=data_cfg["pred_len"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_encoder_layers=cfg["model"].get("n_encoder_layers", 2),
        codebook_K=cb_cfg["K"],
        codebook_tau=cb_cfg["tau"],
        codebook_ema_alpha=cb_cfg["ema_alpha"],
        patch_len=cfg["model"]["patch_len"],
        stride=cfg["model"]["stride"],
        dropout=cfg["model"]["dropout"],
        temporal_config=cfg["model"].get("temporal"),
        use_codebook=cfg["model"].get("use_codebook", True),
        restore_alpha=cfg["model"].get("restore_alpha", 0.1),
        adaptive_alpha=cfg["model"].get("adaptive_alpha", True),
        ts_input=cfg["model"].get("ts_input", False),
        head_type=cfg["model"].get("head_type", "mtgnn"),
    ).to(device)

    # Dummy forward to initialize lazy layers
    with torch.no_grad():
        dummy_x = torch.randn(1, num_variates, data_cfg["seq_len"], device=device)
        dummy_m = torch.ones(1, num_variates, dtype=torch.bool, device=device)
        model(dummy_x, dummy_m)

    ckpt = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    print(f"\n{'='*60}")
    print(f"Running {args.n_samples}-times evaluation")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for i in range(args.n_samples):
        r = evaluate_once(model, test_loader, device, scaler,
                          args.missing_rate, seed=i, num_variates=num_variates)
        results.append(r)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (args.n_samples - i - 1)
            print(f"  [{i+1:3d}/{args.n_samples}] "
                  f"ObsMAE={r['ObsMAE']:.4f} ObsRMSE={r['ObsRMSE']:.4f} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    total_time = time.time() - t0
    mae_list = [r["ObsMAE"] for r in results]
    rmse_list = [r["ObsRMSE"] for r in results]

    print(f"\n{'='*60}")
    print(f"Results ({args.n_samples} samples, {total_time:.0f}s)")
    print(f"{'='*60}")
    print(f"  ObsMAE:  {np.mean(mae_list):.4f} +/- {np.std(mae_list):.4f}")
    print(f"  ObsRMSE: {np.mean(rmse_list):.4f} +/- {np.std(rmse_list):.4f}")

    n_h = len(results[0].get("per_horizon", {}))
    per_horizon_agg = {}
    if n_h > 0:
        print(f"\n  Per-horizon:")
        for t in range(n_h):
            h_maes = [r["per_horizon"][t]["MAE"] for r in results if t in r.get("per_horizon", {})]
            h_rmses = [r["per_horizon"][t]["RMSE"] for r in results if t in r.get("per_horizon", {})]
            per_horizon_agg[t] = {
                "MAE_mean": float(np.mean(h_maes)), "MAE_std": float(np.std(h_maes)),
                "RMSE_mean": float(np.mean(h_rmses)), "RMSE_std": float(np.std(h_rmses)),
            }
            print(f"    h={t+1:2d}: MAE={np.mean(h_maes):.4f}+/-{np.std(h_maes):.4f}  "
                  f"RMSE={np.mean(h_rmses):.4f}+/-{np.std(h_rmses):.4f}")

    out_path = exp_dir / f"eval_{args.n_samples}samples_mr{args.missing_rate}.json"
    save_data = {
        "missing_rate": args.missing_rate,
        "n_samples": args.n_samples,
        "total_time_sec": total_time,
        "ObsMAE_mean": float(np.mean(mae_list)),
        "ObsMAE_std": float(np.std(mae_list)),
        "ObsRMSE_mean": float(np.mean(rmse_list)),
        "ObsRMSE_std": float(np.std(rmse_list)),
        "all_ObsMAE": mae_list,
        "all_ObsRMSE": rmse_list,
    }
    if per_horizon_agg:
        save_data["per_horizon"] = per_horizon_agg

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
