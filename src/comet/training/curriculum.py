"""Three-stage curriculum scheduler for COMET training."""

import math
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class CurriculumState:
    stage: int = 1
    epoch: int = 0
    mask_ratio: float = 0.0
    lambda_align: float = 0.0
    lambda_match: float = 0.0
    consecutive_high_sim: int = 0
    codebook_initialized: bool = False


class CurriculumScheduler:
    """
    Stage 1 (Warm-up):   mask_ratio=0, lambda=0, task loss only.
    Stage 2 (Alignment): mask_ratio and lambda ramp up linearly.
    Stage 3 (Decay):     mask_ratio=max, lambda cosine decays.
    """

    def __init__(
        self,
        stage1_epochs: int = 10,
        stage2_max_epochs: int = 40,
        stage2_min_epochs: int = 20,
        cos_sim_threshold: float = 0.85,
        cos_sim_patience: int = 3,
        lambda_align_max: float = 0.15,
        lambda_match_max: float = 0.075,
        mask_ratio_max: float = 0.75,
        stage3_warmup_epochs: int = 10,
        disable_stage3: bool = False,
    ):
        self.stage1_epochs = stage1_epochs
        self.stage2_max_epochs = stage2_max_epochs
        self.stage2_min_epochs = stage2_min_epochs
        self.cos_sim_threshold = cos_sim_threshold
        self.cos_sim_patience = cos_sim_patience
        self.lambda_align_max = lambda_align_max
        self.lambda_match_max = lambda_match_max
        self.mask_ratio_max = mask_ratio_max
        self.stage3_warmup_epochs = stage3_warmup_epochs
        self.disable_stage3 = disable_stage3
        self.state = CurriculumState()
        self._stage3_start: Optional[int] = None
        self._stage2_progress: float = 0.0

    def step(self, epoch: int,
             val_cos_sim: Optional[float] = None) -> CurriculumState:
        self.state.epoch = epoch

        if epoch <= self.stage1_epochs:
            self.state.stage = 1
            self.state.mask_ratio = 0.0
            self.state.lambda_align = 0.0
            self.state.lambda_match = 0.0
            return self.state

        if self.state.stage == 1:
            self.state.stage = 2
            self.state.consecutive_high_sim = 0

        if self.state.stage == 2:
            s2_epoch = epoch - self.stage1_epochs
            p = min(s2_epoch / max(self.stage2_max_epochs, 1), 1.0)
            self.state.mask_ratio = 0.1 + (self.mask_ratio_max - 0.1) * p
            self.state.lambda_align = self.lambda_align_max * p
            self.state.lambda_match = self.lambda_match_max * p

            if self.disable_stage3:
                return self.state

            if val_cos_sim is not None and s2_epoch >= self.stage2_min_epochs:
                if val_cos_sim > self.cos_sim_threshold:
                    self.state.consecutive_high_sim += 1
                else:
                    self.state.consecutive_high_sim = 0
                if self.state.consecutive_high_sim >= self.cos_sim_patience:
                    self.state.stage = 3
                    self._stage3_start = epoch + 1
                    self._stage2_progress = p

            if s2_epoch >= self.stage2_max_epochs:
                self.state.stage = 3
                self._stage3_start = epoch + 1
                self._stage2_progress = p

            return self.state

        if self.state.stage == 3:
            s3_epoch = epoch - (self._stage3_start or epoch)
            p0 = self._stage2_progress

            if s3_epoch < self.stage3_warmup_epochs:
                frac = s3_epoch / max(self.stage3_warmup_epochs, 1)
                p = p0 + (1.0 - p0) * frac
                self.state.mask_ratio = 0.1 + (self.mask_ratio_max - 0.1) * p
                self.state.lambda_align = self.lambda_align_max * p
                self.state.lambda_match = self.lambda_match_max * p
            else:
                self.state.mask_ratio = self.mask_ratio_max
                decay = min((s3_epoch - self.stage3_warmup_epochs) / max(self.stage2_max_epochs, 1), 1.0)
                self.state.lambda_align = _cosine_decay(self.lambda_align_max, self.lambda_align_max / 10, decay)
                self.state.lambda_match = _cosine_decay(self.lambda_match_max, self.lambda_match_max / 10, decay)

        return self.state

    def should_init_codebook(self, epoch: int) -> bool:
        return epoch == self.stage1_epochs and not self.state.codebook_initialized

    def mark_codebook_initialized(self):
        self.state.codebook_initialized = True


def _cosine_decay(start: float, end: float, progress: float) -> float:
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_masking(num_variates: int, mask_ratio: float,
                  device: torch.device, batch_size: int) -> torch.Tensor:
    """Generate per-sample random observation masks. Returns [B, N] boolean."""
    if mask_ratio <= 0.0 or num_variates <= 1:
        return torch.ones(batch_size, num_variates, dtype=torch.bool, device=device)

    n_miss = max(1, min(int(num_variates * mask_ratio), num_variates - 1))
    rand = torch.rand(batch_size, num_variates, device=device)
    miss_positions = rand.sort(dim=1).indices[:, :n_miss]

    mask = torch.ones(batch_size, num_variates, dtype=torch.bool, device=device)
    mask.scatter_(1, miss_positions, False)
    return mask
