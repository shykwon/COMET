"""Alternative forecast heads: ASTGCN, MSTGCN, TGCN.

Self-contained implementations (no external dependencies).
All use identity adjacency matrix (no pre-computed graph).

References:
  - ASTGCN: Guo et al., "Attention Based Spatial-Temporal Graph
    Convolutional Networks for Traffic Flow Forecasting", AAAI 2019.
  - MSTGCN: Simplified ASTGCN without attention layers.
  - TGCN: Zhao et al., "T-GCN: A Temporal Graph Convolutional Network
    for Traffic Prediction", IEEE TITS 2020.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Chebyshev Utilities (shared by ASTGCN and MSTGCN)
# ---------------------------------------------------------------------------

def _cheb_polynomials_from_identity(N: int, K: int):
    """Chebyshev polynomials from identity adjacency matrix.

    Identity adj -> D = I, L = 0 -> scaled_Laplacian = -I.
    Recurrence: T_0 = I, T_1 = -I, T_k = 2(-I)T_{k-1} - T_{k-2}.
    """
    L_tilde = -np.eye(N, dtype=np.float32)
    polys = [np.eye(N, dtype=np.float32)]
    if K > 1:
        polys.append(L_tilde.copy())
    for i in range(2, K):
        polys.append(2 * L_tilde @ polys[i - 1] - polys[i - 2])
    return polys


# ---------------------------------------------------------------------------
# ASTGCN Internal Components
# ---------------------------------------------------------------------------

class _SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_vertices, num_timesteps):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(num_timesteps))
        self.W2 = nn.Parameter(torch.empty(in_channels, num_timesteps))
        self.W3 = nn.Parameter(torch.empty(in_channels))
        self.bs = nn.Parameter(torch.empty(1, num_vertices, num_vertices))
        self.Vs = nn.Parameter(torch.empty(num_vertices, num_vertices))

    def forward(self, x):
        """x: (B, N, F, T) -> (B, N, N)"""
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (B,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)       # (B,T,N)
        S = torch.matmul(self.Vs, torch.sigmoid(torch.matmul(lhs, rhs) + self.bs))
        return F.softmax(S, dim=1)


class _TemporalAttention(nn.Module):
    def __init__(self, in_channels, num_vertices, num_timesteps):
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(num_vertices))
        self.U2 = nn.Parameter(torch.empty(in_channels, num_vertices))
        self.U3 = nn.Parameter(torch.empty(in_channels))
        self.be = nn.Parameter(torch.empty(1, num_timesteps, num_timesteps))
        self.Ve = nn.Parameter(torch.empty(num_timesteps, num_timesteps))

    def forward(self, x):
        """x: (B, N, F, T) -> (B, T, T)"""
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        E = torch.matmul(self.Ve, torch.sigmoid(torch.matmul(lhs, rhs) + self.be))
        return F.softmax(E, dim=1)


class _ChebConvSAt(nn.Module):
    """Chebyshev graph convolution with spatial attention."""

    def __init__(self, K, in_channels, out_channels):
        super().__init__()
        self.K = K
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)
        ])

    def forward(self, x, cheb_polys, spatial_attention):
        """x: (B, N, F_in, T) -> (B, N, F_out, T)"""
        B, N, _, T = x.shape
        F_out = self.Theta[0].shape[1]
        outputs = []
        for t in range(T):
            signal = x[:, :, :, t]  # (B, N, F_in)
            out = torch.zeros(B, N, F_out, device=x.device, dtype=x.dtype)
            for k in range(self.K):
                T_k_at = cheb_polys[k].mul(spatial_attention)  # (N,N)*(B,N,N) broadcast
                rhs = T_k_at.permute(0, 2, 1).matmul(signal)  # (B,N,F_in)
                out = out + rhs.matmul(self.Theta[k])
            outputs.append(out.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class _ChebConv(nn.Module):
    """Chebyshev graph convolution (no attention)."""

    def __init__(self, K, in_channels, out_channels):
        super().__init__()
        self.K = K
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)
        ])

    def forward(self, x, cheb_polys):
        """x: (B, N, F_in, T) -> (B, N, F_out, T)"""
        B, N, _, T = x.shape
        F_out = self.Theta[0].shape[1]
        outputs = []
        for t in range(T):
            signal = x[:, :, :, t]
            out = torch.zeros(B, N, F_out, device=x.device, dtype=x.dtype)
            for k in range(self.K):
                T_k = cheb_polys[k]  # (N, N)
                rhs = signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                out = out + rhs.matmul(self.Theta[k])
            outputs.append(out.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class _ASTGCNBlock(nn.Module):
    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, num_vertices, num_timesteps):
        super().__init__()
        self.TAt = _TemporalAttention(in_channels, num_vertices, num_timesteps)
        self.SAt = _SpatialAttention(in_channels, num_vertices, num_timesteps)
        self.cheb_conv = _ChebConvSAt(K, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter, nb_time_filter,
            kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter,
            kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, cheb_polys):
        """x: (B, N, F_in, T) -> (B, N, F_out, T')"""
        B, N, C, T = x.shape
        temporal_At = self.TAt(x)  # (B, T, T)
        x_TAt = torch.matmul(
            x.reshape(B, -1, T), temporal_At
        ).reshape(B, N, C, T)
        spatial_At = self.SAt(x_TAt)  # (B, N, N)
        spatial_gcn = self.cheb_conv(x, cheb_polys, spatial_At)  # (B,N,F,T)
        time_out = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (B,F,N,T)
        x_res = self.residual_conv(x.permute(0, 2, 1, 3))          # (B,F,N,T)
        out = self.ln(
            F.relu(x_res + time_out).permute(0, 3, 2, 1)  # (B,T,N,F)
        ).permute(0, 2, 3, 1)  # (B,N,F,T)
        return out


class _MSTGCNBlock(nn.Module):
    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides):
        super().__init__()
        self.cheb_conv = _ChebConv(K, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter, nb_time_filter,
            kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter,
            kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, cheb_polys):
        """x: (B, N, F_in, T) -> (B, N, F_out, T')"""
        spatial_gcn = self.cheb_conv(x, cheb_polys)
        time_out = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        x_res = self.residual_conv(x.permute(0, 2, 1, 3))
        out = self.ln(
            F.relu(x_res + time_out).permute(0, 3, 2, 1)
        ).permute(0, 2, 3, 1)
        return out


# ---------------------------------------------------------------------------
# TGCN Internal Components
# ---------------------------------------------------------------------------

class _TGCNGraphConv(nn.Module):
    """Graph convolution for TGCN using normalized Laplacian with self-loop."""

    def __init__(self, adj_np, num_gru_units, output_dim, bias=0.0):
        super().__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        # Normalized adjacency: D^{-1/2} (A+I) D^{-1/2}
        A = torch.FloatTensor(adj_np) + torch.eye(adj_np.shape[0])
        d_inv_sqrt = torch.pow(A.sum(1), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag(d_inv_sqrt)
        norm_adj = A.matmul(d_mat).T.matmul(d_mat)
        self.register_buffer('laplacian', norm_adj)
        self.weights = nn.Parameter(torch.empty(num_gru_units + 1, output_dim))
        self.biases = nn.Parameter(torch.full((output_dim,), bias))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, inputs, hidden_state):
        """inputs: (B, N), hidden_state: (B, N*H) -> (B, N*O)"""
        B, N = inputs.shape
        H = self._num_gru_units
        inputs = inputs.reshape(B, N, 1)
        hidden_state = hidden_state.reshape(B, N, H)
        concat = torch.cat((inputs, hidden_state), dim=2)  # (B, N, H+1)
        concat = concat.transpose(0, 1).transpose(1, 2).reshape(N, (H + 1) * B)
        a_concat = self.laplacian @ concat  # (N, (H+1)*B)
        a_concat = a_concat.reshape(N, H + 1, B).transpose(0, 2).transpose(1, 2)
        a_concat = a_concat.reshape(B * N, H + 1)
        outputs = a_concat @ self.weights + self.biases  # (B*N, O)
        return outputs.reshape(B, N * self._output_dim)


class _TGCNCell(nn.Module):
    """GRU cell with graph convolution."""

    def __init__(self, adj_np, hidden_dim):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.graph_conv1 = _TGCNGraphConv(adj_np, hidden_dim, hidden_dim * 2, bias=1.0)
        self.graph_conv2 = _TGCNGraphConv(adj_np, hidden_dim, hidden_dim)

    def forward(self, inputs, hidden_state):
        """inputs: (B, N), hidden_state: (B, N*H) -> (B, N*H)"""
        concat = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        r, u = torch.chunk(concat, chunks=2, dim=1)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        return u * hidden_state + (1.0 - u) * c


# ---------------------------------------------------------------------------
# Forecast Head Wrappers
# ---------------------------------------------------------------------------

class ASTGCNHead(nn.Module):
    """ASTGCN forecast head with spatial and temporal attention.

    Default: receives [B, N, L, D] patch embeddings (in_channels=d_model).
    Ablation (ts_input=True): receives [B, N, T] time series (in_channels=1).
    Uses identity adjacency matrix (graph conv = per-node operation).
    """

    def __init__(self, num_variates: int, d_model: int = 128,
                 pred_len: int = 12, seq_len: int = 12,
                 ts_input: bool = False,
                 nb_block: int = 2, K: int = 3,
                 nb_chev_filter: int = 64, nb_time_filter: int = 64,
                 time_strides: int = 1, **kwargs):
        super().__init__()
        self.ts_input = ts_input
        in_channels = 1 if ts_input else d_model

        # Pre-compute Chebyshev polynomials from identity adjacency
        polys_np = _cheb_polynomials_from_identity(num_variates, K)
        for i, p in enumerate(polys_np):
            self.register_buffer(f'_cheb_{i}', torch.from_numpy(p))
        self._K = K

        T_after = seq_len // time_strides
        self.blocks = nn.ModuleList()
        self.blocks.append(_ASTGCNBlock(
            in_channels, K, nb_chev_filter, nb_time_filter,
            time_strides, num_variates, seq_len))
        for _ in range(nb_block - 1):
            self.blocks.append(_ASTGCNBlock(
                nb_time_filter, K, nb_chev_filter, nb_time_filter,
                1, num_variates, T_after))

        self.final_conv = nn.Conv2d(
            T_after, pred_len, kernel_size=(1, nb_time_filter))
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _get_cheb_polys(self):
        return [getattr(self, f'_cheb_{i}') for i in range(self._K)]

    def forward(self, x: torch.Tensor,
                obs_mask=None, restore_alpha=None) -> torch.Tensor:
        """
        Args:
            x: [B, N, L, D] (default) or [B, N, T] (ts_input).
            obs_mask: Ignored (identity adjacency).
            restore_alpha: Ignored.
        Returns:
            [B, N, pred_len].
        """
        if self.ts_input:
            x = x.unsqueeze(2)            # [B,N,T] -> [B,N,1,T]
        else:
            x = x.permute(0, 1, 3, 2)     # [B,N,L,D] -> [B,N,D,L]

        cheb_polys = self._get_cheb_polys()
        for block in self.blocks:
            x = block(x, cheb_polys)

        # (B,N,F,T') -> (B,T',N,F) -> Conv2d -> (B,pred_len,N,1) -> squeeze
        out = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        return out.permute(0, 2, 1)  # (B, N, pred_len)


class MSTGCNHead(nn.Module):
    """MSTGCN forecast head (ASTGCN without attention, simpler and faster).

    Same interface as ASTGCNHead.
    """

    def __init__(self, num_variates: int, d_model: int = 128,
                 pred_len: int = 12, seq_len: int = 12,
                 ts_input: bool = False,
                 nb_block: int = 2, K: int = 3,
                 nb_chev_filter: int = 64, nb_time_filter: int = 64,
                 time_strides: int = 1, **kwargs):
        super().__init__()
        self.ts_input = ts_input
        in_channels = 1 if ts_input else d_model

        polys_np = _cheb_polynomials_from_identity(num_variates, K)
        for i, p in enumerate(polys_np):
            self.register_buffer(f'_cheb_{i}', torch.from_numpy(p))
        self._K = K

        T_after = seq_len // time_strides
        self.blocks = nn.ModuleList()
        self.blocks.append(_MSTGCNBlock(
            in_channels, K, nb_chev_filter, nb_time_filter, time_strides))
        for _ in range(nb_block - 1):
            self.blocks.append(_MSTGCNBlock(
                nb_time_filter, K, nb_chev_filter, nb_time_filter, 1))

        self.final_conv = nn.Conv2d(
            T_after, pred_len, kernel_size=(1, nb_time_filter))
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_cheb_polys(self):
        return [getattr(self, f'_cheb_{i}') for i in range(self._K)]

    def forward(self, x: torch.Tensor,
                obs_mask=None, restore_alpha=None) -> torch.Tensor:
        if self.ts_input:
            x = x.unsqueeze(2)
        else:
            x = x.permute(0, 1, 3, 2)

        cheb_polys = self._get_cheb_polys()
        for block in self.blocks:
            x = block(x, cheb_polys)

        out = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        return out.permute(0, 2, 1)


class TGCNHead(nn.Module):
    """TGCN forecast head (GRU + graph convolution).

    Requires single-channel input. For e2e mode (ts_input=False),
    projects D -> 1 via learned linear layer before feeding to TGCN.
    Uses identity adjacency matrix.
    """

    def __init__(self, num_variates: int, d_model: int = 128,
                 pred_len: int = 12, seq_len: int = 12,
                 ts_input: bool = False, hidden_dim: int = 128,
                 **kwargs):
        super().__init__()
        self.ts_input = ts_input
        self._hidden_dim = hidden_dim
        self.num_variates = num_variates
        self.pred_len = pred_len

        if not ts_input:
            self.channel_proj = nn.Linear(d_model, 1)

        adj = np.eye(num_variates, dtype=np.float32)
        self.tgcn_cell = _TGCNCell(adj, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor,
                obs_mask=None, restore_alpha=None) -> torch.Tensor:
        """
        Args:
            x: [B, N, L, D] (default) or [B, N, T] (ts_input).
            obs_mask: Ignored.
            restore_alpha: Ignored.
        Returns:
            [B, N, pred_len].
        """
        if self.ts_input:
            B, N, T = x.shape
            x_seq = x.permute(0, 2, 1)  # [B, T, N]
        else:
            B, N, L, D = x.shape
            # Project D -> 1: [B,N,L,D] -> [B,N,L,1] -> squeeze -> [B,N,L] -> [B,L,N]
            x_seq = self.channel_proj(x).squeeze(-1).permute(0, 2, 1)

        seq_len = x_seq.shape[1]
        hidden = torch.zeros(
            B, N * self._hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            hidden = self.tgcn_cell(x_seq[:, t, :], hidden)

        # (B, N*H) -> (B, N, H) -> regressor -> (B, N, pred_len)
        output = hidden.reshape(B, N, self._hidden_dim)
        output = self.regressor(output.reshape(-1, self._hidden_dim))
        return output.reshape(B, N, self.pred_len)
