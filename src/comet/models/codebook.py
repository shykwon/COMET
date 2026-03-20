"""Normal-state pattern codebook with EMA updates and dead entry revival."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Codebook(nn.Module):
    """
    Stores K normal-state pattern vectors updated via EMA from teacher signals.

    Args:
        K: Number of codebook entries.
        d: Embedding dimension.
        ema_alpha: EMA decay coefficient.
        tau: Soft lookup temperature.
    """

    def __init__(self, K: int = 128, d: int = 128,
                 ema_alpha: float = 0.99, tau: float = 0.5):
        super().__init__()
        self.K = K
        self.d = d
        self.ema_alpha = ema_alpha
        self.tau = tau

        self.register_buffer("C", torch.randn(K, d) * 0.02)
        self.register_buffer("usage_ema", torch.ones(K) / K)

    def soft_lookup(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            w: Attention weights over codebook [B, K].
        """
        dist_sq = torch.cdist(Q.unsqueeze(0), self.C.unsqueeze(0)).squeeze(0).pow(2)
        return F.softmax(-dist_sq / self.tau, dim=-1)

    def hard_lookup(self, Q: torch.Tensor) -> torch.Tensor:
        """Hard lookup with straight-through estimator.
        Forward: one-hot (argmin). Backward: soft gradient.
        """
        dist_sq = torch.cdist(Q.unsqueeze(0), self.C.unsqueeze(0)).squeeze(0).pow(2)
        w_soft = F.softmax(-dist_sq / self.tau, dim=-1)
        k_star = dist_sq.argmin(dim=-1)
        w_hard = F.one_hot(k_star, self.K).float()
        return w_hard - w_soft.detach() + w_soft

    def perplexity(self, w: torch.Tensor) -> torch.Tensor:
        avg_w = w.detach().mean(dim=0)
        entropy = -(avg_w * torch.log(avg_w + 1e-8)).sum()
        return torch.exp(entropy)

    @torch.no_grad()
    def ema_update(self, Q_full: torch.Tensor, w_full: torch.Tensor,
                   is_daytime: Optional[torch.Tensor] = None,
                   no_revival: bool = False):
        B = w_full.shape[0]
        N_usage = w_full.sum(0)
        self.usage_ema.mul_(0.99).add_(N_usage / B, alpha=0.01)

        new_C = torch.matmul(w_full.T, Q_full) / (N_usage.unsqueeze(1) + 1e-8)
        active = (N_usage > 1e-3).float().unsqueeze(1)
        self.C.copy_(self.ema_alpha * self.C + (1 - self.ema_alpha) * (
            active * new_C + (1 - active) * self.C
        ))
        if not no_revival:
            self._revive_dead_entries(Q_full, is_daytime)

    @torch.no_grad()
    def _revive_dead_entries(self, Q_full: torch.Tensor,
                             is_daytime: Optional[torch.Tensor] = None,
                             threshold: float = 0.01):
        dead = self.usage_ema < threshold
        n_dead = dead.sum().item()
        if n_dead == 0 or n_dead == self.K:
            return

        pool = Q_full
        if is_daytime is not None and is_daytime.any():
            daytime_pool = Q_full[is_daytime]
            if daytime_pool.shape[0] > 0:
                pool = daytime_pool

        idx = torch.randint(0, pool.shape[0], (n_dead,), device=pool.device)
        replacements = pool[idx] + torch.randn(n_dead, self.d, device=pool.device) * pool.std() * 0.1

        dead_idx = dead.nonzero(as_tuple=True)[0]
        self.C[dead_idx] = replacements
        self.usage_ema[dead_idx] = 1.0 / self.K

    @torch.no_grad()
    def init_from_kmeans(self, embeddings: torch.Tensor, n_iter: int = 50,
                         is_daytime: Optional[torch.Tensor] = None):
        if is_daytime is not None and is_daytime.any():
            daytime = embeddings[is_daytime]
            if daytime.shape[0] >= self.K:
                embeddings = daytime

        if embeddings.shape[0] < self.K:
            repeats = (self.K // embeddings.shape[0]) + 1
            embeddings = embeddings.repeat(repeats, 1)[:self.K]
            embeddings = embeddings + torch.randn_like(embeddings) * 0.01

        centroids = embeddings[torch.randperm(embeddings.shape[0], device=embeddings.device)[:self.K]].clone()
        for _ in range(n_iter):
            assignments = torch.cdist(embeddings, centroids).argmin(dim=-1)
            for k in range(self.K):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = embeddings[mask].mean(dim=0)

        self.C.data.copy_(centroids)
        self.usage_ema.fill_(1.0 / self.K)
