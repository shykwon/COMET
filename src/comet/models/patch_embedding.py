"""Patch embedding for time series: (B, N, T) -> (B, N, L, D)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):

    def __init__(self, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.projection = nn.Linear(patch_len, d_model)
        self.position_embedding = None
        self.dropout = nn.Dropout(dropout)

    def get_num_patches(self, seq_len: int) -> int:
        padded = seq_len + self.patch_len - self.stride
        return (padded - self.patch_len) // self.stride + 1

    def _init_position_embedding(self, num_patches: int, device: torch.device):
        if self.position_embedding is not None and self.position_embedding.shape[0] >= num_patches:
            return
        pe = torch.zeros(num_patches, self.d_model, device=device)
        pos = torch.arange(num_patches, dtype=torch.float, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
                        * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.position_embedding = nn.Parameter(pe, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, T) -> (B, N, L, D)"""
        pad_len = self.patch_len - self.stride
        x = F.pad(x, (0, pad_len), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        embeddings = self.projection(patches)
        L = embeddings.shape[2]
        self._init_position_embedding(L, embeddings.device)
        embeddings = embeddings + self.position_embedding[:L].unsqueeze(0).unsqueeze(0)
        return self.dropout(embeddings)
