"""Patch-Level Asymmetric Encoder: observed variate patches → system state Q_sub."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PatchLevelEncoder(nn.Module):
    """
    Patch-level Asymmetric Encoder: compress observed variate patches → Q_sub.

    Architecture:
        1. Flatten observed variate patches: [B, N', L, D] → [B, N'*L, D]
        2. Add Variable ID Embedding (broadcast over L patches)
        3. Prepend [CLS] token
        4. Self-Attention (TransformerEncoder)
        5. CLS output = Q_sub (system state summary)
        6. Remaining tokens reshaped back to [B, N', L, D]
    """

    def __init__(
        self,
        num_variates: int,
        num_patches: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        share_var_id_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.num_variates = num_variates
        self.num_patches = num_patches
        self.d_model = d_model

        self.var_id_embed = share_var_id_embed or nn.Embedding(num_variates, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

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

    def forward(
        self,
        h_patches_obs: torch.Tensor,
        obs_indices: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_patches_obs: Patch embeddings for observed variates [B, N'_max, L, D].
            obs_indices:   Original variate indices [B, N'_max].
            padding_mask:  [B, N'_max] bool, True = padded variate.

        Returns:
            Q_sub:              System state summary [B, D].
            tokens_obs_patches: Enriched observed patch tokens [B, N'_max, L, D].
        """
        B, N_prime, L, D = h_patches_obs.shape

        if obs_indices.dim() == 1:
            obs_indices = obs_indices.unsqueeze(0).expand(B, -1)

        # Add variate ID embedding (broadcast over patches)
        var_id_emb = self.var_id_embed(obs_indices).unsqueeze(2).expand(-1, -1, L, -1)
        tokens = (h_patches_obs + var_id_emb).reshape(B, N_prime * L, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens_with_cls = torch.cat([cls_tokens, tokens], dim=1)

        # Build padding mask: if a variate is padded, all its L patches are masked
        if padding_mask is not None and padding_mask.any():
            patch_padding = padding_mask.unsqueeze(-1).expand(-1, -1, L).reshape(B, N_prime * L)
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=h_patches_obs.device)
            full_padding_mask = torch.cat([cls_pad, patch_padding], dim=1)
        else:
            full_padding_mask = None

        out = self.norm(self.encoder(tokens_with_cls, src_key_padding_mask=full_padding_mask))

        Q_sub = out[:, 0, :]
        tokens_obs_patches = out[:, 1:, :].reshape(B, N_prime, L, D)

        return Q_sub, tokens_obs_patches
