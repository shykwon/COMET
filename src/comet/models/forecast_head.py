"""MTGNN forecast head with dilated inception and bidirectional MixProp."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConstructor(nn.Module):
    """Graph constructor with learned linear transforms and tanh scaling."""

    def __init__(self, num_nodes: int, node_dim: int = 40,
                 subgraph_size: int = 20, alpha: float = 3.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.subgraph_size = min(subgraph_size, num_nodes)
        self.alpha = alpha
        self.emb = nn.Embedding(num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)

    def forward(self) -> torch.Tensor:
        idx = torch.arange(self.num_nodes, device=self.emb.weight.device)
        v1 = torch.tanh(self.alpha * self.lin1(self.emb(idx)))
        v2 = torch.tanh(self.alpha * self.lin2(self.emb(idx)))
        adj = F.relu(torch.tanh(self.alpha * (v1 @ v2.T - v2 @ v1.T)))
        if self.subgraph_size < self.num_nodes:
            _, topk = adj.topk(self.subgraph_size, dim=-1)
            mask = torch.zeros_like(adj)
            mask.scatter_(-1, topk, 1.0)
            adj = adj * mask
        return adj


class MixProp(nn.Module):
    """Mix-hop propagation with residual mixing."""

    def __init__(self, c_in: int, c_out: int, depth: int = 2,
                 alpha: float = 0.05, dropout: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.depth = depth
        self.mlp = nn.Conv1d((depth + 1) * c_in, c_out, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """x: [B, C, N], A: [N, N]."""
        out = [x]
        h = x
        for _ in range(self.depth):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum('bcn,nm->bcm', h, A)
            out.append(h)
        return self.dropout(self.mlp(torch.cat(out, dim=1)))


class DilatedInception(nn.Module):
    """Multi-kernel dilated temporal convolution."""

    def __init__(self, c_in: int, c_out: int, dilation: int = 1,
                 kernel_set=(2, 3, 6, 7)):
        super().__init__()
        n_k = len(kernel_set)
        c_per = c_out // n_k
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_set):
            c = c_per + (c_out - c_per * n_k) if i == n_k - 1 else c_per
            self.convs.append(nn.Conv1d(c_in, c, kernel_size=k, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [conv(x) for conv in self.convs]
        min_len = min(o.shape[-1] for o in outs)
        return torch.cat([o[..., -min_len:] for o in outs], dim=1)


class MTGNNHead(nn.Module):
    """
    MTGNN head with dilated inception and bidirectional graph conv.

    Default: receives patch embeddings [B, N, L, D] (end-to-end, in_dim=d_model).
    Ablation (ts_input=True): receives time series [B, N, T] (in_dim=1).
    """

    def __init__(self, num_variates: int, d_model: int = 128,
                 pred_len: int = 12, seq_len: int = 12,
                 n_layers: int = 3, node_dim: int = 40,
                 subgraph_size: int = 20, conv_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 gcn_depth: int = 2, propalpha: float = 0.05,
                 tanhalpha: float = 3.0, dropout: float = 0.3,
                 dilation_exponential: int = 1,
                 ts_input: bool = False):
        super().__init__()
        self.num_variates = num_variates
        self.pred_len = pred_len
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.conv_channels = conv_channels
        self.ts_input = ts_input

        kernel_set = [2, 3, 6, 7]
        max_k = max(kernel_set)

        if dilation_exponential > 1:
            self.receptive_field = 1 + int(
                (max_k - 1) * (dilation_exponential ** n_layers - 1)
                / (dilation_exponential - 1))
        else:
            self.receptive_field = n_layers * (max_k - 1) + 1

        in_dim = 1 if ts_input else d_model
        padded_len = max(seq_len, self.receptive_field)
        self.padded_len = padded_len

        self.start_conv = nn.Conv1d(in_dim, conv_channels, kernel_size=1)
        self.gc = GraphConstructor(num_variates, node_dim, subgraph_size, tanhalpha)
        self.skip0 = nn.Conv1d(in_dim, skip_channels, kernel_size=padded_len)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gconv_fwd = nn.ModuleList()
        self.gconv_bwd = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        rf = 1
        for i in range(n_layers):
            dilation = dilation_exponential ** i if dilation_exponential > 1 else 1
            self.filter_convs.append(DilatedInception(conv_channels, conv_channels, dilation, kernel_set))
            self.gate_convs.append(DilatedInception(conv_channels, conv_channels, dilation, kernel_set))
            self.gconv_fwd.append(MixProp(conv_channels, conv_channels, gcn_depth, propalpha, dropout))
            self.gconv_bwd.append(MixProp(conv_channels, conv_channels, gcn_depth, propalpha, dropout))
            if dilation_exponential > 1:
                rf = 1 + int((max_k - 1) * (dilation_exponential ** (i + 1) - 1)
                             / (dilation_exponential - 1))
            else:
                rf = (i + 1) * (max_k - 1) + 1
            remaining = max(padded_len - rf + 1, 1)
            self.skip_convs.append(nn.Conv1d(conv_channels, skip_channels, kernel_size=remaining))
            self.norms.append(nn.LayerNorm(conv_channels))

        self.skipE = nn.Conv1d(conv_channels, skip_channels, kernel_size=max(padded_len - rf + 1, 1))
        self.end_conv_1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(end_channels, pred_len, kernel_size=1)

    def get_adj(self) -> torch.Tensor:
        """Extract learned adjacency matrix [N, N] for use by other heads."""
        with torch.no_grad():
            return self.gc().detach().cpu().numpy()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, N, L, D] patch embeddings (default) or [B, N, T] time series (ts_input).
        """
        if self.ts_input:
            B, N, T = x.shape
            h = x.reshape(B * N, 1, T)
            seq_len = T
        else:
            B, N, L, D = x.shape
            h = x.reshape(B * N, L, D).permute(0, 2, 1)  # [B*N, D, L]
            seq_len = L

        if seq_len < self.receptive_field:
            h = F.pad(h, (self.receptive_field - seq_len, 0))

        skip = self.skip0(F.dropout(h, self.dropout_rate, training=self.training))
        h = self.start_conv(h)

        adp = self.gc()
        adp_t = adp.T

        for i in range(self.n_layers):
            residual = h
            h = F.dropout(
                torch.tanh(self.filter_convs[i](h)) * torch.sigmoid(self.gate_convs[i](h)),
                self.dropout_rate, training=self.training)
            skip = skip + self.skip_convs[i](h)

            T_cur, C = h.shape[-1], self.conv_channels
            h_sp = h.view(B, N, C, T_cur).permute(0, 3, 2, 1).reshape(B * T_cur, C, N)
            h_sp = self.gconv_fwd[i](h_sp, adp) + self.gconv_bwd[i](h_sp, adp_t)
            h = (h_sp.view(B, T_cur, C, N).permute(0, 3, 2, 1)
                 .contiguous().reshape(B * N, C, T_cur))
            h = h + residual[..., -T_cur:]
            h = self.norms[i](h.permute(0, 2, 1)).permute(0, 2, 1)

        skip = skip + self.skipE(h)
        h = F.relu(self.end_conv_1(F.relu(skip)))
        return self.end_conv_2(h).squeeze(-1).view(B, N, self.pred_len)
