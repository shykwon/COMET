"""Time series dataset loading and preprocessing."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple


class StandardScaler:
    """Per-variate or global standardization."""

    def __init__(self, global_mode: bool = False):
        self.mean = None
        self.std = None
        self.global_mode = global_mode

    def fit(self, data: np.ndarray):
        if self.global_mode:
            self.mean = data.mean(keepdims=True)
            self.std = data.std(keepdims=True)
        else:
            self.mean = data.mean(axis=0, keepdims=True)
            self.std = data.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset producing (x, y) pairs in (N, T) layout."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int,
                 raw_data: np.ndarray = None):
        self.data = torch.FloatTensor(data)
        self.raw_data = torch.FloatTensor(raw_data) if raw_data is not None else None
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = max(0, len(data) - seq_len - pred_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len].T
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len].T
        if self.raw_data is not None:
            y_raw = self.raw_data[idx + self.seq_len:idx + self.seq_len + self.pred_len].T
            return x, y, y_raw
        return x, y


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    data = df.iloc[:, 1:].values.astype(np.float32)
    for col in range(data.shape[1]):
        bad = data[:, col] < -9000
        if bad.any():
            data[bad, col] = np.median(data[~bad, col])
    return data


def _load_txt(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",").astype(np.float32)


def _load_h5(path: str) -> np.ndarray:
    return pd.read_hdf(path).values.astype(np.float32)


DATASETS = {
    "ETTh1":         ("ETTh1.csv",         _load_csv),
    "ETTm1":         ("ETTm1.csv",         _load_csv),
    "electricity":   ("electricity.txt",   _load_txt),
    "exchange_rate":  ("exchange_rate.txt",  _load_txt),
    "solar":         ("solar.txt",         _load_txt),
    "traffic":       ("traffic.txt",       _load_txt),
    "metr-la":       ("metr-la.h5",        _load_h5),
    "pems-bay":      ("pems-bay.h5",       _load_h5),
    "weather":       ("weather.csv",       _load_csv),
    "ecg5000":       ("ecg5000.txt",       _load_txt),
}

SCALE_FACTORS = {
    "traffic": 1000,
    "ecg5000": 10,
}


def load_raw_data(dataset_name: str, data_dir: str) -> np.ndarray:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASETS.keys())}")

    filename, loader = DATASETS[dataset_name]
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = loader(str(path))
    if dataset_name in SCALE_FACTORS:
        data = data * SCALE_FACTORS[dataset_name]
    return data


def create_dataloaders(
    dataset_name: str,
    data_dir: str,
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    global_scaler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, StandardScaler]:
    """
    Returns:
        (train_loader, val_loader, test_loader, num_variates, scaler)
    """
    data = load_raw_data(dataset_name, data_dir)
    num_variates = data.shape[1]
    print(f"  Loaded {dataset_name}: shape={data.shape}")

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_raw = data[:train_end]
    val_raw = data[train_end:val_end]
    test_raw = data[val_end:]

    scaler = StandardScaler(global_mode=global_scaler)
    scaler.fit(train_raw)

    train_ds = TimeSeriesDataset(scaler.transform(train_raw), seq_len, pred_len, train_raw)
    val_ds = TimeSeriesDataset(scaler.transform(val_raw), seq_len, pred_len, val_raw)
    test_ds = TimeSeriesDataset(scaler.transform(test_raw), seq_len, pred_len, test_raw)
    print(f"  Split: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

    persist = num_workers > 0
    kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persist)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)

    print(f"  num_variates={num_variates}, batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    return train_loader, val_loader, test_loader, num_variates, scaler
