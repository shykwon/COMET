"""Temporal path variants: per-variate independent sequence modeling.

All variants share the same interface: (B, N, L, D) -> (B, N, L, D).
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mamba (default)
# ---------------------------------------------------------------------------

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


class MambaTemporalPath(nn.Module):
    """Per-variate independent Mamba blocks. (B, N, L, D) -> (B, N, L, D)."""

    def __init__(self, d_model: int = 128, n_layers: int = 1,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, **kwargs):
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


# Backward compatibility alias
TemporalPath = MambaTemporalPath


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class TransformerTemporalPath(nn.Module):
    """Per-variate independent Transformer. (B, N, L, D) -> (B, N, L, D).

    PatchEmbedding already provides positional encoding, so no PE here.
    """

    def __init__(self, d_model: int = 128, n_layers: int = 1,
                 n_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, D = x.shape
        x = x.view(B * N, L, D)
        x = self.norm(self.encoder(x))
        return x.view(B, N, L, D)


# ---------------------------------------------------------------------------
# Conv1D (multi-kernel)
# ---------------------------------------------------------------------------

class MultiKernelConv1DBlock(nn.Module):
    """Single block: LayerNorm -> multi-kernel Conv1D -> project -> residual.

    Uses odd kernel sizes with symmetric padding to preserve L exactly.
    """

    def __init__(self, d_model: int, kernels: Tuple[int, ...] = (1, 3, 5),
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding=k // 2)
            for k in kernels
        ])
        self.proj = nn.Linear(d_model * len(kernels), d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*N, L, D] -> [B*N, L, D]"""
        residual = x
        x = self.norm(x)
        x_t = x.transpose(1, 2)                          # [B*N, D, L]
        outs = [conv(x_t).transpose(1, 2) for conv in self.convs]  # each [B*N, L, D]
        x = self.proj(torch.cat(outs, dim=-1))            # [B*N, L, D]
        x = self.dropout(self.act(x))
        return residual + x


class Conv1DTemporalPath(nn.Module):
    """Per-variate independent multi-kernel Conv1D. (B, N, L, D) -> (B, N, L, D)."""

    def __init__(self, d_model: int = 128, n_layers: int = 1,
                 kernels: Tuple[int, ...] = (1, 3, 5),
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiKernelConv1DBlock(d_model, kernels, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, D = x.shape
        x = x.view(B * N, L, D)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).view(B, N, L, D)


# ---------------------------------------------------------------------------
# Identity (no temporal encoding)
# ---------------------------------------------------------------------------

class IdentityTemporalPath(nn.Module):
    """No temporal encoding — just LayerNorm. (B, N, L, D) -> (B, N, L, D)."""

    def __init__(self, d_model: int = 128, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TEMPORAL_REGISTRY = {
    "mamba": MambaTemporalPath,
    "transformer": TransformerTemporalPath,
    "conv1d": Conv1DTemporalPath,
    "identity": IdentityTemporalPath,
}


def create_temporal_path(temporal_type: str = "mamba", **kwargs) -> nn.Module:
    if temporal_type not in TEMPORAL_REGISTRY:
        raise ValueError(
            f"Unknown temporal type '{temporal_type}'. "
            f"Choose from: {list(TEMPORAL_REGISTRY.keys())}"
        )
    return TEMPORAL_REGISTRY[temporal_type](**kwargs)
