"""Loss functions for COMET training."""

import torch
import torch.nn.functional as F


def compute_infonce(Q_sub: torch.Tensor, Q_full: torch.Tensor,
                    tau: float = 0.07) -> torch.Tensor:
    """InfoNCE loss: align Q_sub toward sg(Q_full)."""
    B = Q_sub.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=Q_sub.device, requires_grad=True)
    target = Q_full.detach()
    sim = F.normalize(Q_sub, dim=-1) @ F.normalize(target, dim=-1).T / tau
    return F.cross_entropy(sim, torch.arange(B, device=Q_sub.device))


def compute_kl_match(w_sub: torch.Tensor, w_full: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """KL divergence: KL(w_sub || sg(w_full))."""
    target = w_full.detach().clamp(min=eps)
    w = w_sub.clamp(min=eps)
    return (w * torch.log(w / target)).sum(dim=-1).mean()


def compute_entropy_reg(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative entropy of average weights (minimize to maximize diversity)."""
    avg = w.mean(dim=0)
    return (avg * torch.log(avg + eps)).sum()


def compute_topk_hit_ratio(w_sub: torch.Tensor, w_full: torch.Tensor,
                           k: int = 3) -> float:
    """Fraction of top-k entries shared between student and teacher."""
    with torch.no_grad():
        topk_s = w_sub.topk(k, dim=-1).indices
        topk_t = w_full.detach().topk(k, dim=-1).indices
        hits = sum(
            len(set(topk_s[b].tolist()) & set(topk_t[b].tolist()))
            for b in range(w_sub.shape[0])
        )
        return hits / (w_sub.shape[0] * k)
