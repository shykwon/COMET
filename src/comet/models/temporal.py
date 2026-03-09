"""CI-Mamba temporal path: per-variate independent sequence modeling."""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class SimplifiedSSM(nn.Module):
    """Fallback SSM when mamba-ssm is not installed."""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                              padding=d_conv - 1, groups=d_inner)
        self.ssm_proj = nn.Linear(d_inner, d_state * 2)
        self.ssm_out = nn.Linear(d_state, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        x_main = self.conv(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_main = self.act(x_main)

        ssm = self.ssm_proj(x_main)
        state, gate = ssm.chunk(2, dim=-1)
        ssm_out = self.ssm_out(torch.tanh(state) * torch.sigmoid(gate))

        return self.out_proj(x_main * ssm_out * self.act(z))


class MambaBlock(nn.Module):

    def __init__(self, d_model: int = 128, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.mamba = Mamba2(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        else:
            self.mamba = SimplifiedSSM(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class TemporalPath(nn.Module):
    """Per-variate independent Mamba blocks. (B, N, L, D) -> (B, N, L, D)."""

    def __init__(self, d_model: int = 128, n_layers: int = 1,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, D = x.shape
        x = x.view(B * N, L, D)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).view(B, N, L, D)
