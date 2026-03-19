#!/usr/bin/env python3
"""
Prepare Solar/METR-LA/ECG5000/Traffic data in GinAR format.

GinAR data format (from main_PEM.py):
  - feature_target() creates sliding windows:
    - feature: [n_samples, N, T_in=12]  (input lookback, N variables × T timesteps)
    - target:  [n_samples, N, T_out=12]  (future horizon)
  - Then transposed and expanded:
    - train_x_raw: [n, N, T_in] (NOT transposed in feature_target, already [n, N, T])
    - train_y: [n, T_out, N, 1] (transposed + expand_dims)
    - train_x_mask: [n, T_in, N, 1] (masked version, transposed + expand_dims)
  - In main_ginar.py, concat along dim=-1:
    - train_data = cat([train_x_mask, train_y], dim=-1) → [n, T, N, 2]
    - x = data[:, :, :, 0:in_size]  → [B, T, N, 1] (masked input)
    - y = data[:, :, :, -1]          → [B, T, N] (target = future values)

  GinAR model:
    - Input: [B, H=T_in, N, C=in_size]
    - Output: [B, L=out_len, N]
    - For VSF: H=12 lookback, L=12 horizon

  Masking: variables are zeroed in input, model predicts all N variables' future
"""

import sys
import numpy as np
import pickle
import random
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comet.data.dataset import load_raw_data


def feature_target(data, input_len, output_len):
    """Create sliding windows. Matches GinAR's data/main_PEM.py"""
    fin_feature = []
    fin_target = []
    data_len = data.shape[0]
    for i in range(data_len - input_len - output_len + 1):
        lin_fea_seq = data[i:i+input_len, :]           # [T_in, N]
        lin_tar_seq = data[i+input_len:i+input_len+output_len, :]  # [T_out, N]
        fin_feature.append(lin_fea_seq)
        fin_target.append(lin_tar_seq)
    fin_feature = np.array(fin_feature).transpose((0, 2, 1))  # [n, N, T_in]
    fin_target = np.array(fin_target).transpose((0, 2, 1))    # [n, N, T_out]
    return fin_feature, fin_target


def prepare_dataset(dataset_name, data_dir, output_dir,
                    seq_len=12, pred_len=12, mask_rate=0.85, seed=42):
    """Prepare one dataset in GinAR format."""
    print(f"\n{'='*50}")
    print(f"Preparing {dataset_name}")
    print(f"{'='*50}")

    random.seed(seed)
    np.random.seed(seed)

    # Load raw data
    data = load_raw_data(dataset_name, data_dir)
    T, N = data.shape
    print(f"  Raw shape: ({T}, {N})")

    # Split 70/10/20
    n_train_t = int(T * 0.7)
    n_val_t = int(T * 0.1)

    # Min-max normalization (on full train data)
    max_data = data[:n_train_t].max()
    min_data = data[:n_train_t].min()
    data_norm = (data - min_data) / (max_data - min_data + 1e-8)

    # Create masked data (fixed mask across all samples, like GinAR)
    n_mask = round(N * mask_rate)
    mask_ids = sorted(random.sample(range(N), n_mask))
    data_masked = copy.deepcopy(data_norm)
    data_masked[:, mask_ids] = 0.0

    print(f"  Mask: {n_mask}/{N} variables zeroed (rate={mask_rate})")

    # Create sliding windows from normalized data
    raw_feature, fin_target = feature_target(data_norm, seq_len, pred_len)

    # Create sliding windows from masked data
    mask_feature, _ = feature_target(data_masked, seq_len, pred_len)

    # Total samples
    n_total = raw_feature.shape[0]
    # Split based on time position (matching 70/10/20 of original time series)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)

    # Train
    train_x_raw = raw_feature[:n_train]                     # [n, N, T_in]
    train_y = fin_target[:n_train].transpose(0, 2, 1)       # [n, T_out, N]
    train_y = np.expand_dims(train_y, axis=-1)               # [n, T_out, N, 1]
    train_x_mask = mask_feature[:n_train].transpose(0, 2, 1) # [n, T_in, N]
    train_x_mask = np.expand_dims(train_x_mask, axis=-1)     # [n, T_in, N, 1]

    # Validation
    val_x_raw = raw_feature[n_train:n_train+n_val]
    val_y = fin_target[n_train:n_train+n_val].transpose(0, 2, 1)
    val_y = np.expand_dims(val_y, axis=-1)
    val_x_mask = mask_feature[n_train:n_train+n_val].transpose(0, 2, 1)
    val_x_mask = np.expand_dims(val_x_mask, axis=-1)

    # Test (use raw for test, masking done at eval time)
    test_x_raw = raw_feature[n_train+n_val:]
    test_y = fin_target[n_train+n_val:].transpose(0, 2, 1)
    test_y = np.expand_dims(test_y, axis=-1)
    test_x_mask = mask_feature[n_train+n_val:].transpose(0, 2, 1)
    test_x_mask = np.expand_dims(test_x_mask, axis=-1)

    print(f"  Train: x={train_x_mask.shape}, y={train_y.shape}")
    print(f"  Val:   x={val_x_mask.shape}, y={val_y.shape}")
    print(f"  Test:  x={test_x_mask.shape}, y={test_y.shape}")

    # Save
    out_path = Path(output_dir) / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path / "data.npz",
        train_x_raw=train_x_raw,
        train_y=train_y,
        train_x_mask_85=train_x_mask,
        vail_x_raw=val_x_raw,
        vail_y=val_y,
        vail_x_mask_85=val_x_mask,
        test_x_raw=test_x_raw,
        test_y=test_y,
        test_x_mask_85=test_x_mask,
        max_min=np.array([max_data, min_data]),
        mask_ids=np.array(mask_ids),
    )

    # Identity adj matrix
    adj = np.eye(N, dtype=np.float32)
    with open(out_path / f"adj_{dataset_name}.pkl", "wb") as f:
        pickle.dump(adj, f)

    print(f"  Saved to {out_path}")
    print(f"  max={max_data:.4f}, min={min_data:.4f}")

    return N


if __name__ == "__main__":
    data_dir = str(PROJECT_ROOT / "data" / "raw")
    output_dir = str(Path(__file__).parent / "data")

    for ds_name in ["ecg5000", "solar", "metr-la", "traffic"]:
        try:
            prepare_dataset(ds_name, data_dir, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
