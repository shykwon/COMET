"""
COMET: Codebook-augmented Multivariate time-series forecasting with Expertise Transfer.

End-to-end architecture:
  1. PatchEmbedding: (B, N, T) → (B, N, L, D)
  2. CI-Mamba temporal encoding (per-variate independent)
  3. PatchLevelEncoder: observed variate patches → system state Q_sub + enriched tokens
  4. Codebook soft lookup: Q_sub → normal-state context z_ctx
  5. TwoStageDecoder: missing variate patches restored via obs cross-attn + codebook
  6. MTGNN forecast head: (B, N, L, D) → (B, N, pred_len)

Ablation (ts_input=True):
  Steps 1-5 identical, then:
  6a. Project patch embeddings to time series: (B, N, L, D) → (B, N, T)
  6b. Overwrite observed variates with original time series
  6c. MTGNN head with in_dim=1: (B, N, T) → (B, N, pred_len)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .patch_embedding import PatchEmbedding
from .temporal import create_temporal_path
from .codebook import Codebook
from .encoder import PatchLevelEncoder
from .decoder import TwoStageDecoder
from .forecast_head import MTGNNHead
from .stgcn_heads import ASTGCNHead, MSTGCNHead, TGCNHead

HEAD_CLASSES = {
    'mtgnn': MTGNNHead,
    'astgcn': ASTGCNHead,
    'mstgcn': MSTGCNHead,
    'tgcn': TGCNHead,
}


class COMET(nn.Module):

    def __init__(
        self,
        num_variates: int,
        seq_len: int = 12,
        pred_len: int = 12,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 2,
        codebook_K: int = 16,
        codebook_tau: float = 0.5,
        codebook_ema_alpha: float = 0.99,
        patch_len: int = 4,
        stride: int = 2,
        dropout: float = 0.1,
        temporal_config: Optional[Dict[str, Any]] = None,
        use_codebook: bool = True,
        restore_alpha: float = 0.1,
        adaptive_alpha: bool = True,
        ts_input: bool = False,
        head_type: str = 'mtgnn',
    ):
        super().__init__()
        self.num_variates = num_variates
        self.seq_len = seq_len
        self.use_codebook = use_codebook
        self.restore_alpha = restore_alpha
        self.adaptive_alpha = adaptive_alpha
        self.d_model = d_model
        self.ts_input = ts_input

        # ① Patch Embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len, stride=stride,
            d_model=d_model, dropout=dropout,
        )
        num_patches = self.patch_embedding.get_num_patches(seq_len)
        self.num_patches = num_patches

        # ② Temporal Path (mamba / transformer / conv1d / identity)
        temporal_config = temporal_config or {}
        temporal_type = temporal_config.get("type", "mamba")
        self.temporal_path = create_temporal_path(
            temporal_type=temporal_type,
            d_model=d_model,
            n_layers=temporal_config.get("n_layers", 1),
            d_state=temporal_config.get("d_state", 16),
            d_conv=temporal_config.get("d_conv", 4),
            expand=temporal_config.get("expand", 2),
            n_heads=temporal_config.get("n_heads", 4),
            kernels=tuple(temporal_config.get("kernels", [1, 3, 5])),
            dropout=dropout,
        )

        # Shared Variable ID Embedding
        self.var_id_embed = nn.Embedding(num_variates, d_model)
        nn.init.normal_(self.var_id_embed.weight, std=0.02)

        # ③ Patch-Level Encoder
        self.encoder = PatchLevelEncoder(
            num_variates=num_variates, num_patches=num_patches,
            d_model=d_model, n_heads=n_heads, n_layers=n_encoder_layers,
            dropout=dropout, share_var_id_embed=self.var_id_embed,
        )

        # ④ Codebook
        self.codebook = Codebook(K=codebook_K, d=d_model,
                                 ema_alpha=codebook_ema_alpha, tau=codebook_tau)

        # ⑤ Two-Stage Decoder
        self.decoder = TwoStageDecoder(
            num_variates=num_variates, num_patches=num_patches,
            d_model=d_model, n_heads=n_heads, dropout=dropout,
            share_var_id_embed=self.var_id_embed,
        )

        # ⑥ Forecast Head
        HeadClass = HEAD_CLASSES[head_type]
        if ts_input:
            # Ablation: project patch embeddings → time series, head receives [B,N,T]
            self.patch_to_ts = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.GELU(),
                nn.Linear(d_model * 2, seq_len),
            )
            self.head = HeadClass(
                num_variates=num_variates, d_model=d_model,
                pred_len=pred_len, seq_len=seq_len,
                dropout=0.3, ts_input=True,
            )
        else:
            # Default: head receives patch embeddings [B,N,L,D]
            self.head = HeadClass(
                num_variates=num_variates, d_model=d_model,
                pred_len=pred_len, seq_len=num_patches,
                dropout=0.3, ts_input=False,
            )

    def forward(
        self,
        x_full: torch.Tensor,
        obs_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_full: [B, N, T] all variates (missing will be zero-filled).
            obs_mask: [B, N] boolean, True = observed.

        Returns:
            y_hat [B, N, pred_len], Q_sub [B, d], w_sub [B, K], confidence [B].
        """
        B, N, T = x_full.shape
        D = self.d_model
        L = self.num_patches
        device = x_full.device

        n_obs = obs_mask.sum(dim=1)
        obs_ratio = n_obs.float() / N

        # Zero-fill missing variates
        if not obs_mask.all():
            x_full = x_full.clone()
            x_full[~obs_mask.unsqueeze(-1).expand_as(x_full)] = 0.0

        # ① + ② Patch Embedding + CI-Mamba
        h_patched = self.temporal_path(self.patch_embedding(x_full))  # [B, N, L, D]

        # Build per-sample padded index tensors
        N_prime_max = n_obs.max().item()
        M_max = (N - n_obs).max().item()
        uniform = n_obs.min().item() == N_prime_max

        if uniform and N_prime_max == N:
            obs_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            obs_pad = torch.zeros(B, N, dtype=torch.bool, device=device)
            miss_idx = torch.empty(B, 0, dtype=torch.long, device=device)
            miss_pad = torch.empty(B, 0, dtype=torch.bool, device=device)
            h_obs_patches = h_patched
        else:
            obs_idx = torch.zeros(B, N_prime_max, dtype=torch.long, device=device)
            obs_pad = torch.ones(B, N_prime_max, dtype=torch.bool, device=device)
            miss_idx = torch.zeros(B, max(M_max, 1), dtype=torch.long, device=device) if M_max > 0 \
                else torch.empty(B, 0, dtype=torch.long, device=device)
            miss_pad = torch.ones(B, max(M_max, 1), dtype=torch.bool, device=device) if M_max > 0 \
                else torch.empty(B, 0, dtype=torch.bool, device=device)

            for b in range(B):
                oi = obs_mask[b].nonzero(as_tuple=True)[0]
                mi = (~obs_mask[b]).nonzero(as_tuple=True)[0]
                obs_idx[b, :len(oi)] = oi
                obs_pad[b, :len(oi)] = False
                if len(mi) > 0 and M_max > 0:
                    miss_idx[b, :len(mi)] = mi
                    miss_pad[b, :len(mi)] = False

            # Gather observed variate patches
            idx_4d = obs_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
            h_obs_patches = torch.gather(h_patched, 1, idx_4d)  # [B, N'_max, L, D]

        # ③ PatchLevelEncoder
        Q_sub, tokens_obs_patches = self.encoder(
            h_obs_patches, obs_idx, padding_mask=obs_pad,
        )

        # ④ Codebook
        if self.use_codebook:
            z_ctx, w_sub, confidence = self.codebook.soft_lookup(Q_sub, obs_ratio=obs_ratio)

            # ⑤ TwoStageDecoder
            E_restored = self.decoder(
                h_patches=h_patched,
                tokens_obs_patches=tokens_obs_patches,
                codebook_C=self.codebook.C,
                obs_indices=obs_idx,
                miss_indices=miss_idx,
                obs_padding_mask=obs_pad,
                miss_padding_mask=miss_pad,
                z_ctx=z_ctx,
            )  # [B, N, L, D]
        else:
            K = self.codebook.C.shape[0]
            w_sub = torch.ones(B, K, device=device, dtype=h_patched.dtype) / K
            confidence = torch.zeros(B, device=device, dtype=h_patched.dtype)
            E_restored = h_patched

        # ⑥ Forecast Head
        if self.ts_input:
            # Ablation: pool patches → project to time series → overwrite observed
            x_decoded = self.patch_to_ts(E_restored.mean(dim=2))  # [B, N, L, D] → pool → [B, N, D] → [B, N, T]
            obs_3d = obs_mask.unsqueeze(-1).expand_as(x_decoded)
            x_decoded = x_decoded.clone()
            x_decoded[obs_3d] = x_full[obs_3d].to(x_decoded.dtype)
            head_input = x_decoded  # [B, N, T]
        else:
            head_input = E_restored  # [B, N, L, D]

        if self.use_codebook and self.restore_alpha > 0:
            if self.adaptive_alpha:
                alpha = self.restore_alpha + (1 - self.restore_alpha) * confidence
            else:
                alpha = self.restore_alpha
            y_hat = self.head(head_input, obs_mask=obs_mask, restore_alpha=alpha)
        else:
            y_hat = self.head(head_input)

        if return_embeddings:
            E_var = E_restored.mean(dim=2)  # [B, N, D] for compatibility
            return y_hat, Q_sub, w_sub, confidence, E_var, None
        return y_hat, Q_sub, w_sub, confidence

    def forward_full(self, x: torch.Tensor,
                     return_embeddings: bool = False):
        """Teacher forward with all variates observed.

        Only runs up to encoder + codebook (skips decoder and head)
        since only Q_full and w_full are needed for alignment losses.
        """
        B, N, T = x.shape
        device = x.device

        # ① + ② Patch Embedding + Temporal
        h_patched = self.temporal_path(self.patch_embedding(x))  # [B, N, L, D]

        # ③ PatchLevelEncoder (all variates observed)
        obs_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        obs_pad = torch.zeros(B, N, dtype=torch.bool, device=device)
        Q_full, tokens_obs_patches = self.encoder(h_patched, obs_idx, padding_mask=obs_pad)

        # ④ Codebook
        if self.use_codebook:
            _, w_full, _ = self.codebook.soft_lookup(Q_full, obs_ratio=torch.ones(B, device=device))
        else:
            K = self.codebook.C.shape[0]
            w_full = torch.ones(B, K, device=device, dtype=h_patched.dtype) / K

        if return_embeddings:
            E_var = h_patched.mean(dim=2)  # [B, N, D]
            return None, Q_full, w_full, E_var
        return None, Q_full, w_full
