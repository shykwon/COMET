#!/usr/bin/env python3
"""Training script for COMET."""

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
from comet.training.losses import (
    compute_infonce, compute_kl_match, compute_entropy_reg, compute_topk_hit_ratio,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/default.yaml"))
    p.add_argument("--dataset", type=str)
    p.add_argument("--data_dir", type=str)
    p.add_argument("--pred_len", type=int)
    p.add_argument("--seq_len", type=int)
    p.add_argument("--d_model", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--missing_rate", type=float)
    p.add_argument("--codebook_K", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--loss_fn", type=str, choices=["mse", "mae", "huber"])
    p.add_argument("--ts_input", action="store_true",
                   help="Ablation: feed time series to head instead of embeddings")
    p.add_argument("--head_type", type=str, choices=["mtgnn", "astgcn", "mstgcn", "tgcn"],
                   help="Forecast head type")
    p.add_argument("--log_dir", type=str)
    p.add_argument("--n_encoder_layers", type=int)
    p.add_argument("--entropy_reg_weight", type=float)
    p.add_argument("--miss_loss_weight", type=float)
    p.add_argument("--stage1_epochs", type=int)
    p.add_argument("--stage2_min_epochs", type=int)
    p.add_argument("--stage2_max_epochs", type=int)
    p.add_argument("--restore_alpha", type=float)
    p.add_argument("--disable_stage3", action="store_true")
    p.add_argument("--no_codebook", action="store_true")
    p.add_argument("--temporal_type", type=str, choices=["mamba", "transformer", "conv1d", "identity"],
                   help="Ablation: temporal path variant")
    p.add_argument("--null_val", type=float)
    p.add_argument("--amp_bf16", action="store_true")
    p.add_argument("--resume", type=str)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    overrides = {
        "dataset": ("data", "dataset"),
        "data_dir": ("data", "data_dir"),
        "pred_len": ("data", "pred_len"),
        "seq_len": ("data", "seq_len"),
        "d_model": ("model", "d_model"),
        "batch_size": ("training", "batch_size"),
        "epochs": ("training", "epochs"),
        "patience": ("training", "patience"),
        "seed": ("training", "seed"),
        "lr": ("training", "learning_rate"),
        "log_dir": ("logging", "log_dir"),
        "loss_fn": ("training", "loss_fn"),
        "n_encoder_layers": ("model", "n_encoder_layers"),
        "entropy_reg_weight": ("training", "entropy_reg_weight"),
        "miss_loss_weight": ("training", "miss_loss_weight"),
        "stage1_epochs": ("training", "stage1_epochs"),
        "stage2_min_epochs": ("training", "stage2_min_epochs"),
        "stage2_max_epochs": ("training", "stage2_max_epochs"),
        "restore_alpha": ("model", "restore_alpha"),
    }
    for arg_name, (section, key) in overrides.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            cfg[section][key] = val

    if args.missing_rate is not None:
        cfg["training"]["missing_rate"] = args.missing_rate
        cfg["training"]["mask_ratio_max"] = args.missing_rate
    if args.codebook_K:
        cfg["model"]["codebook"]["K"] = args.codebook_K
    if args.disable_stage3:
        cfg["training"]["disable_stage3"] = True
    if args.no_codebook:
        cfg["model"]["use_codebook"] = False
    if args.null_val is not None:
        cfg["training"]["null_val"] = args.null_val
    if args.ts_input:
        cfg["model"]["ts_input"] = True
    if args.head_type:
        cfg["model"]["head_type"] = args.head_type
    if args.temporal_type:
        cfg["model"]["temporal"]["type"] = args.temporal_type
    if args.debug:
        cfg["training"]["epochs"] = 3
        cfg["training"]["stage1_epochs"] = 1
        cfg["training"]["stage2_max_epochs"] = 1

    return cfg


class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, metric, epoch):
        if metric < self.best:
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        self.best = float("inf")
        self.counter = 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, scaler, missing_rate=0.0,
             loss_fn="mse", deterministic_seed=None):
    model.eval()
    total_loss = 0.0
    preds_all, trues_all, masks_all, raw_all = [], [], [], []
    n_batches = 0

    if deterministic_seed is not None:
        rng_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        torch.manual_seed(deterministic_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(deterministic_seed)

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            y_raw = batch[2].to(device) if len(batch) > 2 else None
            B, N, T = x.shape

            obs_mask = apply_masking(N, missing_rate, device, B)
            y_hat, _, _, _ = model(x, obs_mask)

            if missing_rate > 0:
                m3d = obs_mask.unsqueeze(-1).expand_as(y_hat)
                y_h, y_t = y_hat[m3d], y[m3d]
            else:
                y_h, y_t = y_hat, y

            if loss_fn == "mae":
                loss = F.l1_loss(y_h, y_t)
            elif loss_fn == "huber":
                loss = F.huber_loss(y_h, y_t)
            else:
                loss = F.mse_loss(y_h, y_t)

            total_loss += loss.item()
            n_batches += 1
            preds_all.append(y_hat.cpu().numpy())
            trues_all.append(y.cpu().numpy())
            if y_raw is not None:
                raw_all.append(y_raw.cpu().numpy())
            if missing_rate > 0:
                masks_all.append(obs_mask.cpu().numpy())

    if deterministic_seed is not None:
        torch.random.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

    preds = np.concatenate(preds_all)
    trues = np.concatenate(trues_all)
    shape = preds.shape

    preds_flat = preds.transpose(0, 2, 1).reshape(-1, shape[1])
    trues_flat = trues.transpose(0, 2, 1).reshape(-1, shape[1])
    preds_inv = scaler.inverse_transform(preds_flat).reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)
    trues_inv = scaler.inverse_transform(trues_flat).reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)

    if raw_all:
        nonnull = np.concatenate(raw_all) != 0.0
    else:
        nonnull = np.abs(trues_inv) > 1e-4
    nonnull_count = max(nonnull.sum(), 1)

    result = {
        "loss": total_loss / max(n_batches, 1),
        "MAE": float(np.abs(preds_inv - trues_inv)[nonnull].sum() / nonnull_count),
        "RMSE": float(np.sqrt(((preds_inv - trues_inv) ** 2)[nonnull].sum() / nonnull_count)),
    }

    if masks_all:
        masks = np.expand_dims(np.concatenate(masks_all), -1).repeat(shape[2], axis=-1)
        nonnull_f = nonnull.astype(float)
        obs_valid = masks * nonnull_f
        obs_err = np.abs(preds_inv - trues_inv) * obs_valid
        obs_sq = ((preds_inv - trues_inv) ** 2) * obs_valid
        result["ObsMAE"] = float(obs_err.sum() / max(obs_valid.sum(), 1))
        result["ObsRMSE"] = float(np.sqrt(obs_sq.sum() / max(obs_valid.sum(), 1)))
        miss_valid = (1.0 - masks) * nonnull_f
        result["MissMAE"] = float((np.abs(preds_inv - trues_inv) * miss_valid).sum() / max(miss_valid.sum(), 1))

    return result


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, state,
                    scaler_amp, loss_fn, entropy_reg_weight, miss_loss_weight,
                    collect_q, q_buffer, daytime_buffer, log_interval,
                    scaler, denorm_loss, null_val, disable_ema):
    model.train()
    totals = {k: 0.0 for k in ["loss", "task", "nce", "match", "entropy", "miss"]}
    cos_sims, perplexities = [], []
    n_batches = 0

    mask_ratio = state.mask_ratio
    lam_a = state.lambda_align
    lam_m = state.lambda_match

    if hasattr(model, 'codebook'):
        model.codebook._revive_count = 0

    if scaler is not None:
        sc_std = torch.tensor(scaler.std.squeeze(), dtype=torch.float32, device=device)
        sc_mean = torch.tensor(scaler.mean.squeeze(), dtype=torch.float32, device=device)
    else:
        sc_std = sc_mean = None

    for batch_idx, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)
        y_raw = batch[2].to(device) if len(batch) > 2 else None
        B, N, T = x.shape

        obs_mask = apply_masking(N, mask_ratio, device, B)

        with torch.amp.autocast('cuda', enabled=scaler_amp is not None, dtype=torch.bfloat16):
            y_hat, Q_sub, w_sub, confidence = model(x, obs_mask)

            need_teacher = lam_a > 0 or lam_m > 0
            if need_teacher:
                with torch.no_grad():
                    _, Q_full, w_full = model.forward_full(x)
            else:
                Q_full = Q_sub.detach()
                w_full = w_sub.detach()

            y_hat_l, y_l = y_hat, y

            if denorm_loss and sc_std is not None:
                std_e = sc_std.unsqueeze(0).unsqueeze(-1)
                mean_e = sc_mean.unsqueeze(0).unsqueeze(-1)
                y_hat_l = y_hat_l * std_e + mean_e
                y_l = y_l * std_e + mean_e

            if loss_fn == "huber":
                abs_e = torch.abs(y_hat_l - y_l)
                errors = torch.where(abs_e <= 1.0, 0.5 * (y_hat_l - y_l).pow(2), abs_e - 0.5)
            elif loss_fn == "mae":
                errors = torch.abs(y_hat_l - y_l)
            else:
                errors = (y_hat_l - y_l).pow(2)

            nv_mask = None
            if null_val is not None:
                if y_raw is not None:
                    nv_mask = y_raw != null_val
                else:
                    nv_mask = y_l.abs() > 1e-4

            L_miss = torch.tensor(0.0, device=device)
            if mask_ratio > 0:
                obs_3d = obs_mask.unsqueeze(-1).expand_as(y_hat_l)
                obs_f = obs_3d.float()
                if nv_mask is not None:
                    obs_f = obs_f * nv_mask.float()
                L_task = (errors * obs_f).sum() / obs_f.sum().clamp(min=1.0)

                if miss_loss_weight > 0:
                    miss_f = (~obs_3d).float()
                    if nv_mask is not None:
                        miss_f = miss_f * nv_mask.float()
                    L_miss = (errors * miss_f).sum() / miss_f.sum().clamp(min=1.0)
                    L_task = L_task + miss_loss_weight * L_miss
            elif nv_mask is not None:
                nv_f = nv_mask.float()
                L_task = (errors * nv_f).sum() / nv_f.sum().clamp(min=1.0)
            else:
                L_task = errors.mean()

            L_nce = compute_infonce(Q_sub, Q_full) if lam_a > 0 else torch.tensor(0.0, device=device)
            L_match = compute_kl_match(w_sub, w_full) if lam_m > 0 else torch.tensor(0.0, device=device)

            if state.codebook_initialized and entropy_reg_weight > 0:
                L_ent = compute_entropy_reg(w_sub)
            else:
                L_ent = torch.tensor(0.0, device=device)

            loss = L_task + lam_a * L_nce + lam_m * L_match + entropy_reg_weight * L_ent

        optimizer.zero_grad(set_to_none=True)
        if scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Daytime detection for codebook filtering
        is_daytime = None
        if sc_std is not None:
            with torch.no_grad():
                y_orig = y * sc_std.unsqueeze(0).unsqueeze(-1) + sc_mean.unsqueeze(0).unsqueeze(-1)
                is_daytime = (y_orig.abs() > 0.1).float().mean(dim=[1, 2]) > 0.3

        if not disable_ema and state.codebook_initialized and (lam_a > 0 or lam_m > 0):
            model.codebook.ema_update(Q_full.detach(), w_full.detach(), is_daytime=is_daytime)

        if collect_q and q_buffer is not None:
            with torch.no_grad():
                q_buffer.append(Q_full.detach().cpu())
                if daytime_buffer is not None and is_daytime is not None:
                    daytime_buffer.append(is_daytime.detach().cpu())

        totals["loss"] += loss.item()
        totals["task"] += L_task.item()
        totals["nce"] += L_nce.item()
        totals["match"] += L_match.item()
        totals["entropy"] += L_ent.item()
        totals["miss"] += L_miss.item()
        n_batches += 1

        with torch.no_grad():
            cos_sims.append(F.cosine_similarity(Q_sub, Q_full.detach(), dim=-1).mean().item())
            perplexities.append(model.codebook.perplexity(w_sub).item())

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f} task={L_task.item():.4f} "
                  f"nce={L_nce.item():.4f} match={L_match.item():.4f} "
                  f"ent={L_ent.item():.4f} cos={cos_sims[-1]:.4f} ppl={perplexities[-1]:.1f}")

    n = max(n_batches, 1)
    return {k: v / n for k, v in totals.items()} | {
        "cos_sim": np.mean(cos_sims) if cos_sims else 0.0,
        "perplexity": np.mean(perplexities) if perplexities else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    set_seed(train_cfg["seed"])
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, num_variates, scaler = create_dataloaders(
        dataset_name=data_cfg["dataset"],
        data_dir=data_cfg["data_dir"],
        seq_len=data_cfg["seq_len"],
        pred_len=data_cfg["pred_len"],
        batch_size=train_cfg["batch_size"],
        num_workers=cfg["hardware"].get("num_workers", 4),
        pin_memory=cfg["hardware"].get("pin_memory", True),
        global_scaler=data_cfg.get("global_scaler", False),
    )

    cb_cfg = model_cfg["codebook"]
    ts_input = model_cfg.get("ts_input", False)
    head_type = model_cfg.get("head_type", "mtgnn")
    temporal_type = model_cfg.get("temporal", {}).get("type", "mamba")
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
        restore_alpha=model_cfg.get("restore_alpha", 0.1),
        adaptive_alpha=model_cfg.get("adaptive_alpha", True),
        ts_input=ts_input,
        head_type=head_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: COMET ({n_params:,} params, {n_params/1e6:.2f}M)")
    print(f"  head={head_type}, temporal={temporal_type}, Codebook K={cb_cfg['K']}, ts_input={ts_input}, "
          f"use_codebook={model_cfg.get('use_codebook', True)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    total_steps = train_cfg["epochs"] * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    scaler_amp = None
    if args.amp_bf16 and device.type == "cuda":
        scaler_amp = torch.amp.GradScaler('cuda', enabled=False)

    curriculum = CurriculumScheduler(
        stage1_epochs=train_cfg.get("stage1_epochs", 10),
        stage2_max_epochs=train_cfg.get("stage2_max_epochs", 40),
        stage2_min_epochs=train_cfg.get("stage2_min_epochs", 20),
        cos_sim_threshold=train_cfg.get("cos_sim_threshold", 0.85),
        cos_sim_patience=train_cfg.get("cos_sim_patience", 3),
        lambda_align_max=train_cfg.get("lambda_align_max", 0.15),
        lambda_match_max=train_cfg.get("lambda_match_max", 0.075),
        mask_ratio_max=train_cfg.get("mask_ratio_max", 0.75),
        disable_stage3=train_cfg.get("disable_stage3", False),
    )

    missing_rate = train_cfg.get("missing_rate", 0.75)
    loss_fn = train_cfg.get("loss_fn", "mse")
    entropy_reg = train_cfg.get("entropy_reg_weight", 0.01)
    miss_loss_w = train_cfg.get("miss_loss_weight", 0.0)
    disable_ema = train_cfg.get("disable_ema", False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    use_cb = model_cfg.get("use_codebook", True)
    cb_tag = "" if use_cb else "_nocb"
    head_tag = f"_{head_type}" if head_type != "mtgnn" else ""
    exp_name = f"comet_{data_cfg['dataset']}_K{cb_cfg['K']}_{temporal_type}{head_tag}{cb_tag}_s{train_cfg['seed']}_{timestamp}"
    log_dir = Path(cfg["logging"]["log_dir"]) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\nExperiment: {exp_name}")
    print(f"  Missing rate: {missing_rate}, Loss: {loss_fn}, Epochs: {train_cfg['epochs']}")

    early_stop = EarlyStopping(patience=train_cfg.get("patience", 30))
    best_miss_mae = float("inf")
    train_log = []
    q_buffer, daytime_buffer = [], []
    prev_stage = 0

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch}")

    print(f"\n{'='*60}\nTraining\n{'='*60}")

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        t0 = time.time()

        cos_sim = train_log[-1]["cos_sim"] if train_log else None
        state = curriculum.step(epoch, val_cos_sim=cos_sim)

        if state.stage >= 2 and prev_stage < 2:
            early_stop.reset()
            best_miss_mae = float("inf")
        prev_stage = state.stage

        if curriculum.should_init_codebook(epoch) and q_buffer:
            Q_all = torch.cat(q_buffer).to(device)
            is_day = torch.cat(daytime_buffer).to(device) if daytime_buffer else None
            model.codebook.init_from_kmeans(Q_all, n_iter=50, is_daytime=is_day)
            with torch.no_grad():
                sample = Q_all[:min(512, len(Q_all))]
                w_test = F.softmax(-torch.cdist(sample, model.codebook.C).pow(2) / model.codebook.tau, dim=-1)
                print(f"  [Codebook] K-Means init, perplexity={model.codebook.perplexity(w_test).item():.1f}")
            q_buffer.clear()
            daytime_buffer.clear()
            curriculum.mark_codebook_initialized()

        collect = state.stage == 1 and not state.codebook_initialized

        print(f"\nEpoch {epoch}/{train_cfg['epochs']} | Stage {state.stage} | "
              f"mask={state.mask_ratio:.2f} | lam_a={state.lambda_align:.3f}")

        metrics = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler, device, state,
            scaler_amp, loss_fn, entropy_reg, miss_loss_w,
            collect, q_buffer if collect else None,
            daytime_buffer if collect else None,
            cfg["logging"].get("log_interval", 50),
            scaler, train_cfg.get("denorm_loss", False),
            train_cfg.get("null_val"), disable_ema,
        )

        val_oracle = evaluate(model, val_loader, device, scaler, 0.0, loss_fn)
        val_miss = evaluate(model, val_loader, device, scaler, missing_rate, loss_fn, deterministic_seed=7777)
        dt = time.time() - t0

        obs_mae = val_miss.get("ObsMAE", val_miss["MAE"])
        print(f"  oracle={val_oracle['MAE']:.4f} | obs={obs_mae:.4f} | "
              f"cos={metrics['cos_sim']:.4f} | ppl={metrics['perplexity']:.1f} | {dt:.0f}s")

        train_log.append({
            "epoch": epoch, "stage": state.stage,
            "train_loss": metrics["loss"], "cos_sim": metrics["cos_sim"],
            "perplexity": metrics["perplexity"],
            "val_oracle_MAE": val_oracle["MAE"], "val_obs_MAE": obs_mae,
        })

        if state.stage == 1:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(), "config": cfg},
                       log_dir / "best_oracle_model.pt")

        if state.stage >= 2:
            if obs_mae < best_miss_mae:
                best_miss_mae = obs_mae
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                             "optimizer_state_dict": optimizer.state_dict(), "config": cfg},
                           log_dir / "best_model.pt")
                print(f"  ** Best (ObsMAE={best_miss_mae:.4f})")

            if early_stop.step(obs_mae, epoch):
                print(f"\n  Early stopping at epoch {epoch} (best={early_stop.best_epoch})")
                break

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "config": cfg},
                   log_dir / "latest.pt")

    # Final evaluation
    print(f"\n{'='*60}\nFinal Evaluation\n{'='*60}")

    best_path = log_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model (epoch {ckpt['epoch']})")

    test_result = evaluate(model, test_loader, device, scaler, missing_rate, loss_fn, deterministic_seed=9999)
    print(f"  ObsMAE={test_result.get('ObsMAE', 0):.4f}, "
          f"ObsRMSE={test_result.get('ObsRMSE', 0):.4f}, "
          f"MissMAE={test_result.get('MissMAE', 0):.4f}")

    with open(log_dir / "results.json", "w") as f:
        json.dump({"test": test_result, "config": cfg}, f, indent=2, default=str)
    with open(log_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nSaved to {log_dir}")


if __name__ == "__main__":
    main()
