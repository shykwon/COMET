"""Two-Stage Restoration Decoder: patch-level cross-attention for missing variates."""

import torch
import torch.nn as nn
from typing import Optional


class TwoStageDecoder(nn.Module):
    """
    Two-stage patch-level restoration decoder.

    Missing variates:
      Stage A: cross-attend to observed variate patches (temporal transfer)
      Stage B: cross-attend to codebook entries (normal-state prior)

    Observed variates:
      Codebook refinement cross-attention + residual from original patches.

    All operations at patch level [B, N, L, D].
    """

    def __init__(
        self,
        num_variates: int,
        num_patches: int,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        share_var_id_embed: Optional[nn.Embedding] = None,
        use_film: bool = False,
    ):
        super().__init__()
        self.num_variates = num_variates
        self.num_patches = num_patches
        self.d_model = d_model
        self.use_film = use_film

        self.var_id_embed = share_var_id_embed or nn.Embedding(num_variates, d_model)

        self.mask_embed = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.normal_(self.mask_embed, std=0.02)

        self.patch_pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, d_model))
        nn.init.normal_(self.patch_pos_embed, std=0.02)

        # Stage A: obs→miss cross-attention
        self.cross_attn_obs = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

        # Stage B: codebook cross-attention
        self.cross_attn_cb = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_b1 = nn.LayerNorm(d_model)
        self.norm_b2 = nn.LayerNorm(d_model)
        self.ffn_b = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

        # FiLM: alternative to Stage B cross-attention
        if use_film:
            self.film_gamma = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.film_beta = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.film_norm = nn.LayerNorm(d_model)

        # Observed: codebook refinement
        self.cross_attn_obs_refine = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_obs1 = nn.LayerNorm(d_model)
        self.norm_obs2 = nn.LayerNorm(d_model)
        self.ffn_obs = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(
        self,
        h_patches: torch.Tensor,
        tokens_obs_patches: torch.Tensor,
        codebook_C: torch.Tensor,
        obs_indices: torch.Tensor,
        miss_indices: torch.Tensor,
        obs_padding_mask: Optional[torch.Tensor] = None,
        miss_padding_mask: Optional[torch.Tensor] = None,
        w_sub: Optional[torch.Tensor] = None,
        skip_codebook: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            h_patches:          CI-Mamba patch embeddings for ALL variates [B, N, L, D].
            tokens_obs_patches: Encoder-enriched observed patch tokens [B, N'_max, L, D].
            codebook_C:         Codebook entries [K, D].
            obs_indices:        Observed variate indices [B, N'_max].
            miss_indices:       Missing variate indices [B, M_max].
            obs_padding_mask:   [B, N'_max] bool, True = padded.
            miss_padding_mask:  [B, M_max] bool, True = padded.
            w_sub:              Codebook attention weights [B, K] from soft lookup.
                                When provided, gates C_expanded so that cross-attention
                                sees w_sub-scaled codebook entries.

        Returns:
            E_restored: Restored patch embeddings [B, N, L, D].
        """
        B, N, L, D = h_patches.shape
        device = h_patches.device
        _dtype = h_patches.dtype

        if obs_indices.dim() == 1:
            obs_indices = obs_indices.unsqueeze(0).expand(B, -1)
        if miss_indices.dim() == 1:
            miss_indices = miss_indices.unsqueeze(0).expand(B, -1)

        N_prime_max = obs_indices.shape[1]
        M_max = miss_indices.shape[1]

        _no_obs_pad = obs_padding_mask is None or not obs_padding_mask.any().item()
        _no_miss_pad = miss_padding_mask is None or not miss_padding_mask.any().item()

        C_expanded = codebook_C.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # Codebook gating: scale entries by w_sub so cross-attention
        # focuses on the codebook entries that Q_sub selected.
        if w_sub is not None:
            C_expanded = w_sub.unsqueeze(-1) * C_expanded  # [B, K, D]

        # Fast path: all variates observed
        if N_prime_max == N and M_max == 0 and _no_obs_pad:
            obs_flat = tokens_obs_patches.reshape(B, N * L, D)
            if not skip_codebook:
                if self.use_film and w_sub is not None:
                    z_ctx = torch.matmul(w_sub, codebook_C)
                    gamma = self.film_gamma(z_ctx)
                    beta = self.film_beta(z_ctx)
                    obs_normed = self.film_norm(obs_flat)
                    obs_out = obs_flat + gamma.unsqueeze(1) * obs_normed + beta.unsqueeze(1)
                else:
                    attn_obs, _ = self.cross_attn_obs_refine(
                        query=self.norm_obs1(obs_flat), key=C_expanded, value=C_expanded,
                    )
                    obs_out = obs_flat + attn_obs
                    obs_out = obs_out + self.ffn_obs(self.norm_obs2(obs_out))
                obs_out = h_patches.reshape(B, N * L, D) + obs_out
            else:
                obs_out = h_patches.reshape(B, N * L, D) + obs_flat
            return obs_out.reshape(B, N, L, D)

        E_restored = torch.zeros(B, N, L, D, device=device, dtype=_dtype)

        # ==== Observed variates ====
        if N_prime_max > 0:
            idx_4d = obs_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
            h_obs_patches = torch.gather(h_patches, 1, idx_4d)

            obs_flat = tokens_obs_patches.reshape(B, N_prime_max * L, D)
            if not skip_codebook:
                if self.use_film and w_sub is not None:
                    z_ctx = torch.matmul(w_sub, codebook_C)
                    gamma = self.film_gamma(z_ctx)
                    beta = self.film_beta(z_ctx)
                    obs_normed = self.film_norm(obs_flat)
                    obs_out = obs_flat + gamma.unsqueeze(1) * obs_normed + beta.unsqueeze(1)
                    obs_out = h_obs_patches.reshape(B, N_prime_max * L, D) + obs_out
                else:
                    attn_obs, _ = self.cross_attn_obs_refine(
                        query=self.norm_obs1(obs_flat), key=C_expanded, value=C_expanded,
                    )
                    obs_out = obs_flat + attn_obs
                    obs_out = obs_out + self.ffn_obs(self.norm_obs2(obs_out))
                obs_out = h_obs_patches.reshape(B, N_prime_max * L, D) + obs_out
            else:
                obs_out = h_obs_patches.reshape(B, N_prime_max * L, D) + obs_flat
            obs_out = obs_out.reshape(B, N_prime_max, L, D)

            if _no_obs_pad:
                idx_scatter = obs_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
                E_restored.scatter_(1, idx_scatter, obs_out.to(_dtype))
            else:
                for b in range(B):
                    valid = ~obs_padding_mask[b]
                    E_restored[b, obs_indices[b][valid]] = obs_out[b][valid].to(_dtype)

        # ==== Missing variates ====
        if M_max > 0:
            v_miss = self.var_id_embed(miss_indices).unsqueeze(2).expand(-1, -1, L, -1)
            patch_pos = self.patch_pos_embed[:, :, :L, :]
            miss_tokens = self.mask_embed.expand(B, M_max, L, -1) + v_miss + patch_pos
            miss_flat = miss_tokens.reshape(B, M_max * L, D)

            if miss_padding_mask is not None and miss_padding_mask.any():
                miss_q_pad = miss_padding_mask.unsqueeze(-1).expand(-1, -1, L).reshape(B, M_max * L)
            else:
                miss_q_pad = None

            # Stage A: cross-attend to observed patches
            obs_kv = tokens_obs_patches.reshape(B, N_prime_max * L, D)
            if obs_padding_mask is not None and obs_padding_mask.any():
                obs_kv_pad = obs_padding_mask.unsqueeze(-1).expand(-1, -1, L).reshape(B, N_prime_max * L)
            else:
                obs_kv_pad = None

            miss_a, _ = self.cross_attn_obs(
                query=self.norm_a1(miss_flat), key=obs_kv, value=obs_kv,
                key_padding_mask=obs_kv_pad,
            )
            miss_flat = miss_flat + miss_a
            miss_flat = miss_flat + self.ffn_a(self.norm_a2(miss_flat))

            # Stage B: codebook → cross-attention or FiLM (skip if no codebook)
            if not skip_codebook:
                if self.use_film and w_sub is not None:
                    # FiLM: w_sub @ C → context → γ, β → modulate
                    z_ctx = torch.matmul(w_sub, codebook_C)  # [B, D]
                    gamma = self.film_gamma(z_ctx)  # [B, D]
                    beta = self.film_beta(z_ctx)    # [B, D]
                    miss_normed = self.film_norm(miss_flat)
                    miss_flat = miss_flat + gamma.unsqueeze(1) * miss_normed + beta.unsqueeze(1)
                else:
                    miss_b, _ = self.cross_attn_cb(
                        query=self.norm_b1(miss_flat), key=C_expanded, value=C_expanded,
                    )
                    miss_flat = miss_flat + miss_b
                    miss_flat = miss_flat + self.ffn_b(self.norm_b2(miss_flat))

            miss_out = miss_flat.reshape(B, M_max, L, D)

            if _no_miss_pad:
                idx_scatter = miss_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
                E_restored.scatter_(1, idx_scatter, miss_out.to(_dtype))
            else:
                for b in range(B):
                    valid = ~miss_padding_mask[b]
                    E_restored[b, miss_indices[b][valid]] = miss_out[b][valid].to(_dtype)

        return E_restored
