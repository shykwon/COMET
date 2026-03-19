"""
VIDA E2E Patch: Minimal modifications to make VIDA end-to-end.

Changes:
(A) Joint training: unfreeze forecaster, add to optimizer
(B) Embedding input: TCN keeps temporal dim, skip decoder, MTGNN in_dim changed

Usage:
  1. Copy this file to VSF_Unified/external/vida-vsf/VIDA/
  2. Patch the original files (see apply_patch() below)
  3. Run with: python main_vida.py --e2e_mode ...
"""

# =============================================================
# PATCH 1: TCN - keep temporal dimension
# =============================================================
# In models/vida.py, class TCN, forward():
#
# BEFORE (line 320):
#   out = out_1[:, :, -1]       # [B, final_out_channels, T] → [B, final_out_channels]
#   return out                   # [B, final_out_channels]
#
# AFTER:
#   return out_1                 # [B, final_out_channels, T]  (keep temporal dim)


# =============================================================
# PATCH 2: Encoder - output with temporal dimension
# =============================================================
# In models/vida.py, class tf_encoder, forward():
#
# BEFORE (lines 91-98):
#   x = x.reshape(batch, in_d * num_nodes, seq_len)
#   ef, out_ft = self.freq_feature(x)
#   ef = F.relu(self.bn_freq(self.avg(ef).squeeze()))  # [B, 2*modes]
#   et = self.tcn(x)                                     # [B, final_out_channels]
#   f = torch.concat([ef, et], -1)                       # [B, 2*modes + final_out_channels]
#   f = F.normalize(f)
#   return f, out_ft
#
# AFTER (e2e mode):
#   x = x.reshape(batch, in_d * num_nodes, seq_len)
#   et = self.tcn(x)                                     # [B, final_out_channels, T]
#   return et   # skip freq features, keep temporal dim
#               # shape: [B, final_out_channels, T]


# =============================================================
# PATCH 3: Skip decoder, feed TCN output to MTGNN
# =============================================================
# In trainer.py, pretrain() and alignment():
#
# BEFORE:
#   feat_src, out_s = self.encoder(input)
#   src_recons = self.decoder(feat_src, out_s)          # [B, 1, N, T]
#   src_recons[:, :, idx_subset, :] = input[:, :, idx_subset, :]
#   pred_result = self.forecaster(src_recons, ...)
#
# AFTER (e2e mode):
#   feat_src = self.encoder(input)                       # [B, final_out_channels, T]
#   # Reshape for MTGNN: [B, final_out_channels, T] → [B, final_out_channels, N, T]
#   # Note: encoder processes all N variables together (flattened to channels)
#   # We need per-variable features for MTGNN
#   # feat_src is [B, C, T] where C=final_out_channels
#   # MTGNN expects [B, in_dim, N, T]
#   # → expand: [B, C, 1, T] → repeat → [B, C, N, T]? No, this loses per-variable info
#
#   # Actually, the TCN input is [B, N, T] (variables as channels)
#   # TCN block1: Conv1d(N, mid_channels, T) → [B, mid_channels, T]
#   # This mixes all variables into mid_channels — per-variable info is already mixed
#
#   # Better approach: use TCN block1 output BEFORE variable mixing
#   # Or: use conv_block1 with per-variable processing


# =============================================================
# PROBLEM: VIDA's TCN mixes variables in the first layer
# =============================================================
#
# TCN input: [B, N=137, T=12]  (all variables as channels)
# conv_block1: Conv1d(N=137, mid_channels=512, kernel=17)
#   → This MIXES all 137 variables into 512 channels in the first operation
#   → Individual variable information is lost
#   → Output [B, 512, T] has no per-variable structure
#
# MTGNN expects: [B, in_dim, N, T]  (per-variable features)
#   → Needs to know which features belong to which variable
#
# This is a FUNDAMENTAL incompatibility:
#   VIDA: variable-mixing encoder (all variables → shared features)
#   MTGNN: variable-preserving forecaster (per-variable graph conv)
#
# Possible solutions:
# (a) Apply TCN per-variable (channel-independent), then stack
#     → Changes VIDA's architecture significantly
# (b) Add a projection layer: [B, mid_channels, T] → [B, in_dim, N, T]
#     → Requires learning to un-mix variables
# (c) Use only the decoder output reshaped differently
#     → Still goes through time-series reconstruction


def analyze_vida_incompatibility():
    """Print analysis of why VIDA can't trivially become e2e."""
    print("""
    ===================================================
    VIDA E2E Incompatibility Analysis
    ===================================================

    VIDA's encoder fundamentally MIXES variables:
      Input:  [B, N, T]     (137 variables, 12 timesteps)
      TCN L1: Conv1d(137 → 512)  → all variables mixed
      TCN L2: Conv1d(512 → 12)   → further compressed
      Output: [B, 12]       (single vector, no variable/temporal dim)

    MTGNN expects per-variable features:
      Input:  [B, in_dim, N, T]  (features PER variable)
      GCN operates on N dimension (variable graph)

    COMET preserves per-variable structure:
      Input:  [B, N, T]
      Patch:  [B, N, L, D]   (per-variable patches)
      Enc:    [B, N', L, D]  (per-variable, observed only)
      Dec:    [B, N, L, D]   (per-variable, restored)
      Head:   [B, N, L, D] → MTGNN(in_dim=D)

    Key insight: COMET's channel-independent design preserves
    per-variable structure throughout, enabling direct embedding
    input to graph-based forecasters. VIDA's variable-mixing
    encoder destroys this structure in the first layer.

    This is NOT just an engineering limitation — it's a
    fundamental architectural difference that makes e2e
    embedding input possible in COMET but not in VIDA.
    ===================================================
    """)


if __name__ == "__main__":
    analyze_vida_incompatibility()
