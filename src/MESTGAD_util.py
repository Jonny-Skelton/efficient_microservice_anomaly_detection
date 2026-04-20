"""
MESTGAD_util.py — Utility modules for MESTGAD

Extensions over MSTGAD:
  1. MambaTemporalModule replaces Temporal_Attention (TAM)
     - Separate Mamba SSM block per modality (node, trace, log)
     - Lightweight linear cross-modal mixing layer post-scan
     - Linear O(W) complexity vs quadratic O(W^2) in original TAM
  2. AssociationDiscrepancy layer on top of SAM
     - Extracts GAT attention coefficients as "series association"
     - Learns a prior distribution over expected attention patterns
     - KL(prior || series) as an anomaly signal that supplements
       reconstruction error

Modules carried over unchanged from MSTGAD_util.py:
  - FeedForward, AddNorm, AddALL, FFN
  - Spatial_Attention (SAM) — modified only to *return* attention weights
  - Encoder_Decoder_Attention (CAM)
  - Embed
  - adj2adj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse, remove_self_loops
from mamba_ssm import Mamba
import math


# =============================================================================
# Unchanged utility functions / modules from MSTGAD
# =============================================================================

def adj2adj(graph, batch_size, window_size, zdim):
    """Convert dense adjacency to sparse edge indices for both node-graph
    and edge-graph (line graph) representations.

    Returns:
        node_adj:   (2, num_node_edges)  — sparse COO indices for node graph
        node_efea:  (N, N, zdim)         — edge feature mask (broadcast)
        edge_adj:   (2, num_edge_edges)  — sparse COO indices for line graph
        edge_efea:  (num_edge_edges,)    — which original node each edge maps to
    """
    graph1 = graph.squeeze(0).squeeze(0).repeat(batch_size, window_size, 1, 1) \
        .reshape(-1, graph.shape[-2], graph.shape[-1])
    adj0, adj1, fea = [], [], []
    node_adj = dense_to_sparse(graph1)[0]
    node_efea = graph.unsqueeze(-1).repeat(1, 1, zdim)
    for num in range(node_adj.shape[1]):
        idx = torch.argwhere(node_adj[1] == num)
        idy = torch.argwhere(node_adj[0] == num)
        adj0.append(idx.repeat(1, idy.shape[0]).reshape(-1))
        adj1.append(idy.repeat(idx.shape[0], 1).reshape(-1))
        fea.append(torch.ones(
            idy.shape[0] * idx.shape[0], device=graph.device) * num)

    adj = torch.stack([torch.concat(adj0), torch.concat(adj1)], dim=0)
    fea = torch.concat(fea)
    edge_adj, edge_efea = remove_self_loops(adj, fea)
    return node_adj, node_efea, edge_adj, edge_efea


class FeedForward(nn.Module):
    def __init__(self, node_embedding_dim, FeedForward_dim, Dropout):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(node_embedding_dim, FeedForward_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(Dropout),
            nn.Linear(FeedForward_dim, node_embedding_dim)
        )

    def forward(self, X):
        return self.ff(X)


class AddNorm(nn.Module):
    def __init__(self, node_embedding_dim, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(node_embedding_dim)

    def forward(self, X_old, X):
        return self.ln(self.dropout(X) + X_old)


class AddALL(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, dropout=0.1):
        super().__init__()
        self.addnorm_node = AddNorm(node_embedding_dim, dropout)
        self.addnorm_trace = AddNorm(edge_embedding_dim, dropout)
        self.addnorm_log = AddNorm(log_embedding_dim, dropout)

    def forward(self, node_old, trace_old, log_old, x_node, x_trace, x_log):
        return self.addnorm_node(node_old, x_node.reshape(node_old.shape)), \
            self.addnorm_trace(trace_old, x_trace.reshape(trace_old.shape)), \
            self.addnorm_log(log_old, x_log.reshape(log_old.shape))


class FFN(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.ff_node = FeedForward(node_embedding_dim, node_embedding_dim * 4, dropout)
        self.addnorm_node = AddNorm(node_embedding_dim, dropout)
        self.ff_trace = FeedForward(edge_embedding_dim, edge_embedding_dim * 4, dropout)
        self.addnorm_trace = AddNorm(edge_embedding_dim, dropout)
        self.ff_log = FeedForward(log_embedding_dim, log_embedding_dim * 4, dropout)
        self.addnorm_log = AddNorm(log_embedding_dim, dropout)

    def forward(self, x_node, x_trace, x_log):
        return self.addnorm_node(x_node, self.ff_node(x_node)), \
            self.addnorm_trace(x_trace, self.ff_trace(x_trace)), \
            self.addnorm_log(x_log, self.ff_log(x_log))


class Embed(nn.Module):
    """Input embedding + sinusoidal positional encoding along the time axis.

    For node/log data (dim=4):  input (B, W, N, F_raw) → output (B, W, N, D)
    For edge data     (dim=5):  input (B, W, N, N, F_raw) → output (B, W, N, N, D)

    Returns:
        X_enc:     embedded input with positional encoding (encoder input)
        X_shifted: right-shifted version with PE (decoder input)
    """
    def __init__(self, raw_dim, embedding_dim, max_len=1000, dim=4):
        super(Embed, self).__init__()
        self.linear = nn.Linear(raw_dim, embedding_dim)
        self.dim = dim
        pe = torch.zeros((1, max_len, embedding_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim)
        pe[:, :, 0::2] = torch.sin(X)
        pe[:, :, 1::2] = torch.cos(X)
        if dim == 4:
            pe = pe.unsqueeze(2)
        elif dim == 5:
            pe = pe.unsqueeze(2).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = self.linear(X)
        if self.dim == 4:
            padding = (0, 0, 0, 0, 1, 0)
            X_new = F.pad(X, padding, "constant", 0)
            return X + Variable(self.pe[:, :X.shape[1], :, :], requires_grad=False), \
                   X_new[:, :X.shape[1], :, :] + Variable(self.pe[:, :X.shape[1], :, :], requires_grad=False)
        else:
            padding = (0, 0, 0, 0, 0, 0, 1, 0)
            X_new = F.pad(X, padding, "constant", 0)
            return X + Variable(self.pe[:, :X.shape[1], :, :, :], requires_grad=False), \
                   X_new[:, :X.shape[1], :, :, :] + Variable(self.pe[:, :X.shape[1], :, :, :], requires_grad=False)


# =============================================================================
# Cross Attention Module (CAM) — unchanged from MSTGAD
# =============================================================================

class Encoder_Decoder_Attention(nn.Module):
    """CAM: Cross Attention Module.

    Used only in the decoder to attend from decoder hidden states to
    encoder output Z_t.  Standard multi-head cross-attention per modality.
    """
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim,
                 heads_node=4, heads_edge=4, heads_log=4, dropout=0.1):
        super(Encoder_Decoder_Attention, self).__init__()
        self.attention_node = nn.MultiheadAttention(
            embed_dim=node_embedding_dim, num_heads=heads_node, batch_first=True, dropout=dropout)
        self.attention_trace = nn.MultiheadAttention(
            embed_dim=edge_embedding_dim, num_heads=heads_edge, batch_first=True, dropout=dropout)
        self.attention_log = nn.MultiheadAttention(
            embed_dim=log_embedding_dim, num_heads=heads_log, batch_first=True, dropout=dropout)

    def forward(self, x_node, x_trace, x_log, z_node, z_trace, z_log):
        x_node = x_node.reshape(x_node.shape[0], -1, x_node.shape[-1])
        z_node = z_node.reshape(z_node.shape[0], -1, z_node.shape[-1])
        x_node = self.attention_node(x_node, z_node, z_node)[0]

        x_trace = x_trace.reshape(x_trace.shape[0], -1, x_trace.shape[-1])
        z_trace = z_trace.reshape(z_trace.shape[0], -1, z_trace.shape[-1])
        x_trace = self.attention_trace(x_trace, z_trace, z_trace)[0]

        x_log = x_log.reshape(x_log.shape[0], -1, x_log.shape[-1])
        z_log = z_log.reshape(z_log.shape[0], -1, z_log.shape[-1])
        x_log = self.attention_log(x_log, z_log, z_log)[0]

        return x_node, x_trace, x_log


# =============================================================================
# NEW: Spatial Attention Module (SAM) — modified to expose attention weights
# =============================================================================

class Spatial_Attention(nn.Module):
    """SAM: Spatial Attention Module (graph attention over the MST graph).

    Two-layer GAT that alternates between:
      1. node2node: updates node features using edge features
      2. edge2node: updates edge features using (updated) node features

    MESTGAD MODIFICATION:
      The forward method now returns attention coefficients alongside the
      updated features.  GATv2Conv supports `return_attention_weights=True`
      which gives back (edge_index, alpha) where alpha are the softmax
      attention coefficients per edge.

      These coefficients are the "series association" used by the
      AssociationDiscrepancy module.
    """
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim,
                 heads_n2e=4, heads_e2n=4, dropout=0.1, batch_size=10,
                 window_size=16):
        super(Spatial_Attention, self).__init__()
        self.batch_size = batch_size
        self.window_size = window_size

        self.node2node = GATv2Conv(
            in_channels=node_embedding_dim + log_embedding_dim,
            out_channels=int((node_embedding_dim + log_embedding_dim) / heads_n2e),
            heads=heads_n2e, dropout=dropout, edge_dim=edge_embedding_dim,
            add_self_loops=False)
        self.egde2node = GATv2Conv(
            in_channels=edge_embedding_dim,
            out_channels=int(edge_embedding_dim / heads_e2n),
            heads=heads_e2n, dropout=dropout,
            edge_dim=node_embedding_dim + log_embedding_dim,
            add_self_loops=False)

    def forward(self, x_node, x_trace, x_log, node_adj, edge_adj, edge_efea,
                return_attention_weights=False):
        """
        Args:
            x_node:  (B, W, N, D_node)
            x_trace: (B, W, E, D_edge)   — E = number of edges in graph
            x_log:   (B, W, N, D_log)
            node_adj:  (2, num_edges_batched) — sparse COO for node graph
            edge_adj:  (2, num_edges_batched) — sparse COO for line graph
            edge_efea: (num_edges_batched,)   — node index per line-graph edge
            return_attention_weights: if True, also return GAT alpha values

        Returns:
            x_node, x_trace, x_log  — updated features (same shapes as input)

            If return_attention_weights=True, additionally returns:
            node_attn_weights: attention coefficients from the node2node GAT
                               shape: (num_edges_batched, heads_n2e)
            edge_attn_weights: attention coefficients from the edge2node GAT
                               shape: (num_line_edges_batched, heads_e2n)
        """
        D_node_orig = x_node.shape[-1]
        node = torch.concat([x_node, x_log], dim=-1)
        node = node.reshape(-1, node.shape[-1])
        x_trace = x_trace.reshape(-1, x_trace.shape[-1])

        if return_attention_weights:
            node, (_, node_alpha) = self.node2node(
                node, node_adj, x_trace, return_attention_weights=True)
            x_trace, (_, edge_alpha) = self.egde2node(
                x_trace, edge_adj, node[edge_efea.long()], return_attention_weights=True)
        else:
            node = self.node2node(node, node_adj, x_trace)
            x_trace = self.egde2node(x_trace, edge_adj, node[edge_efea.long()])
            node_alpha = edge_alpha = None

        # Reshape back to (B, W, N/E, D)
        x_node = node[:, :D_node_orig].reshape(
            self.batch_size, self.window_size, -1, D_node_orig)
        x_trace = x_trace.reshape(
            self.batch_size, self.window_size, -1, x_trace.shape[-1])
        x_log = node[:, D_node_orig:].reshape(
            self.batch_size, self.window_size, -1, x_log.shape[-1])

        if return_attention_weights:
            return x_node, x_trace, x_log, node_alpha, edge_alpha

        return x_node, x_trace, x_log


# =============================================================================
# NEW: Mamba Selective SSM Block
# =============================================================================
class MambaBlock(nn.Module):
    """Thin wrapper around the fused `mamba_ssm.Mamba` kernel.
 
    The previous pure-PyTorch implementation (Python `for t in range(W)`
    loop over the selective scan) is functionally correct but scales
    ~quadratically in wall-clock because:
      1. Each timestep launches its own set of GPU kernels via the Python
         interpreter (O(W) launch overhead, each with fixed latency).
      2. Autograd saves the hidden state at each step for the backward
         pass; per-step memory and bookkeeping costs grow with W.
 
    `mamba_ssm.Mamba` fuses the entire scan into a single CUDA op using the
    parallel associative scan from the Mamba paper (Gu & Dao 2023). The
    forward and backward passes each run in O(W) with a small constant,
    matching the theoretical complexity.
 
    Interface:
        Input:  (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)
 
    Args:
        d_model:    input/output feature dimension
        d_state:    SSM hidden state dimension (default 16, per Mamba paper)
        d_conv:     local 1-D conv kernel width (default 4)
        expand:     expansion factor for the inner dimension (default 2)
 
    Note on hardware requirements:
        `mamba_ssm.Mamba` requires a CUDA-capable GPU and the fused kernel
        extension (`causal_conv1d` + `selective_scan_cuda`) to be installed.
        See the project handoff notes for the install recipe. Falling back
        to a CPU forward is not supported by the fused kernel.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
 
        # use_fast_path=False: the fast path calls causal_conv1d_cuda directly with a
        # 7-arg signature that the installed causal_conv1d 1.1.3 .so does not support.
        # The slow path uses the 5-arg Python wrapper which matches the installed .so.
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
 
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)  — must be on a CUDA device
        Returns:
            y: (batch, seq_len, d_model)
        """
        return self.mamba(x)


class old_MambaBlock(nn.Module):
    """Single Mamba selective state space model block.

    Processes a sequence of length W (window size) with linear O(W) complexity.
    Replaces one head of the original multi-head temporal self-attention.

    Architecture (per the Mamba paper, Gu & Dao 2023):
        x → linear_in → [x_branch, z_branch]
        x_branch → conv1d → SiLU → SSM(delta, B, C) → output
        z_branch → SiLU → gate
        output = output * gate → linear_out

    The SSM is the selective scan:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C_t * h_t

    Where A_bar, B_bar are discretized from continuous parameters using
    the input-dependent step size delta_t.

    Args:
        d_model:    input/output feature dimension
        d_state:    SSM hidden state dimension (N in the Mamba paper, e.g. 16)
        d_conv:     local convolution kernel width (e.g. 4)
        expand:     expansion factor for inner dimension (e.g. 2)

    Input shape:  (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # TODO: Define the following layers:
        #
        # 1. self.in_proj — Linear(d_model, 2 * d_inner)
        #    Projects input into x_branch and z_branch (split in forward).
        #

        self.in_proj = nn.Linear(d_model, 2*self.d_inner)

        # 2. self.conv1d — Conv1d(d_inner, d_inner, kernel_size=d_conv,
        #                         groups=d_inner, padding=d_conv-1)
        #    Depthwise conv for local context on the x_branch.
        #    NOTE: groups=d_inner makes this depthwise (each channel convolved
        #    independently). The padding=d_conv-1 is "causal-style" — you'll
        #    need to slice off the extra timesteps in forward to keep length W.
        #

        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv-1)

        # 3. SSM parameter projections (all from d_inner):
        #    self.x_proj — Linear(d_inner, d_state + d_state + 1)
        #    This single projection produces B_t, C_t, and delta_t for each
        #    timestep. You'll split its output in forward:
        #      B_t:     (batch, seq_len, d_state)  — input-dependent B
        #      C_t:     (batch, seq_len, d_state)  — input-dependent C
        #      delta_t: (batch, seq_len, 1)        — step size (before softplus)
        #

        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1)

        # 4. self.delta_proj — Linear(1, d_inner)
        #    Broadcasts the scalar delta_t to match d_inner for discretization.
        #    Alternatively, you can make x_proj output d_inner instead of 1 for
        #    delta and skip this layer — design choice.
        #

        self.delta_proj = nn.Linear(1, self.d_inner)

        # 5. self.A_log — nn.Parameter(torch.log(A_init))
        #    The continuous state matrix A, stored in log-space for stability.
        #    Shape: (d_inner, d_state)
        #    Initialize A_init as a diagonal-like structure:
        #      A_init = repeat(arange(1, d_state+1), 'd -> e d', e=d_inner)
        #    Then store log(A_init) as the parameter.
        #    In forward, recover A = -exp(A_log) (negative for stability).
        #
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A_init))

        # 6. self.D — nn.Parameter(torch.ones(d_inner))
        #    The skip/residual connection parameter (D in the SSM formulation).
        #

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 7. self.out_proj — Linear(d_inner, d_model)
        #    Final projection back to d_model.

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        B, W, _ = x.shape

        xz = self.in_proj(x)                          # (B, W, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)      # each (B, W, d_inner)

        # Depthwise conv along time, causal (trim right padding)
        x_branch = x_branch.transpose(1, 2)           # (B, d_inner, W)
        x_branch = self.conv1d(x_branch)[:, :, :W]    # (B, d_inner, W)
        x_branch = x_branch.transpose(1, 2)           # (B, W, d_inner)
        x_branch = F.silu(x_branch)

        # Input-dependent SSM parameters
        x_ssm = self.x_proj(x_branch)                 # (B, W, 2*d_state + 1)
        B_t = x_ssm[:, :, :self.d_state]              # (B, W, d_state)
        C_t = x_ssm[:, :, self.d_state:2*self.d_state]  # (B, W, d_state)
        delta_raw = x_ssm[:, :, -1:]                  # (B, W, 1)
        delta = F.softplus(self.delta_proj(delta_raw)) # (B, W, d_inner)

        # Discretize A and B (zero-order hold)
        A = -torch.exp(self.A_log)                     # (d_inner, d_state)
        delta_A = delta.unsqueeze(-1) * A[None, None]  # (B, W, d_inner, d_state)
        A_bar = torch.exp(delta_A)                     # (B, W, d_inner, d_state)
        # B_t: (B, W, d_state) → unsqueeze(2) → (B, W, 1, d_state)
        # delta: (B, W, d_inner) → unsqueeze(-1) → (B, W, d_inner, 1)
        B_bar = delta.unsqueeze(-1) * B_t.unsqueeze(2)  # (B, W, d_inner, d_state)

        # Selective scan (sequential recurrence, O(W))
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(W):
            h = A_bar[:, t] * h + B_bar[:, t] * x_branch[:, t].unsqueeze(-1)
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)
        y_ssm = torch.stack(ys, dim=1)                # (B, W, d_inner)
        y_ssm = y_ssm + x_branch * self.D             # skip connection

        # Gating
        z = F.silu(z_branch)                          # (B, W, d_inner)
        output = y_ssm * z                            # (B, W, d_inner)

        return self.out_proj(output)                   # (B, W, d_model)


# =============================================================================
# NEW: Mamba Temporal Module — replaces Temporal_Attention (TAM)
# =============================================================================

class MambaTemporalModule(nn.Module):
    """Replaces Temporal_Attention with per-modality Mamba blocks + cross-modal mixing.

    Design rationale:
      - Original TAM runs separate nn.MultiheadAttention per modality, then
        fuses via the shared attention score matrix C (average of per-modality
        QK^T / sqrt(d)).
      - MESTGAD runs separate MambaBlock per modality, then fuses via a
        lightweight linear cross-modal mixing layer that projects the
        concatenated modality outputs and redistributes.

    The cross-modal mixing replaces C's role: it lets each modality's temporal
    representation be informed by the others' temporal patterns, but through a
    learned linear map rather than an attention-score average.

    Shape contract (must match Temporal_Attention's interface):
        Input:  x_node  (B, W, N, D_node)
                x_trace (B, W, E, D_edge)
                x_log   (B, W, N, D_log)
        Output: same shapes as input

    Args:
        node_embedding_dim, edge_embedding_dim, log_embedding_dim: feature dims
        d_state:     Mamba SSM hidden state dimension
        d_conv:      Mamba local conv kernel width
        expand:      Mamba expansion factor
        dropout:     dropout rate for the mixing layer
        window_size: W
        batch_size:  B
    """
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim,
                 trace2pod,  # kept in signature for drop-in compatibility; unused here
                 d_state=16, d_conv=4, expand=2, dropout=0.1,
                 window_size=16, batch_size=10):
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size

        # TODO: Define three MambaBlock instances, one per modality.
        #
        # self.mamba_node  = MambaBlock(d_model=node_embedding_dim,
        #                               d_state=d_state, d_conv=d_conv, expand=expand)
        # self.mamba_trace = MambaBlock(d_model=edge_embedding_dim, ...)
        # self.mamba_log   = MambaBlock(d_model=log_embedding_dim, ...)

        # TODO: Define the cross-modal mixing layer.
        #
        # Strategy: After per-modality Mamba, for each node/edge entity,
        # you have three temporal output vectors. Concatenate along the
        # feature dim, project down, then split back.
        #
        # But node has N entities while trace has E entities, so direct
        # concat isn't trivial. Two options:
        #
        # Option A (simpler): Mix node and log only (they share the N-entity
        #   axis), and leave trace separate. This is actually close to what
        #   the original C matrix does — the trace2pod mapping bridges them.
        #
        #   self.mix_node_log = nn.Sequential(
        #       nn.Linear(node_embedding_dim + log_embedding_dim,
        #                 node_embedding_dim + log_embedding_dim),
        #       nn.SiLU(),
        #       nn.Dropout(dropout),
        #   )
        #   Then split the output back into node and log portions.
        #
        # Option B (richer): Also project trace through trace2pod to get a
        #   per-node trace summary, concatenate all three, mix, then
        #   inverse-project the trace portion back. More faithful to the
        #   C matrix's cross-modal role.
        #
        # Option A implemented below.

        self.mamba_node = MambaBlock(d_model=node_embedding_dim,
                                     d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_trace = MambaBlock(d_model=edge_embedding_dim,
                                      d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_log = MambaBlock(d_model=log_embedding_dim,
                                    d_state=d_state, d_conv=d_conv, expand=expand)

        self.node_dim = node_embedding_dim
        self.log_dim = log_embedding_dim

        # Cross-modal mixing: node and log share the N-entity axis
        self.mix_node_log = nn.Sequential(
            nn.Linear(node_embedding_dim + log_embedding_dim,
                      node_embedding_dim + log_embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_node, x_trace, x_log, mask=False):
        """
        NOTE ON THE `mask` PARAMETER:
          In original TAM, mask=True triggers causal masking for the decoder.
          Mamba is *inherently causal* — the recurrence only looks backward.
          So mask=True requires NO special handling; the same Mamba forward
          pass is used in both encoder and decoder. This is one of the nice
          properties of replacing TAM with Mamba.

          You can simply ignore the mask parameter, or add an assertion:
            assert not mask or True, "Mamba is inherently causal"

        IMPLEMENTATION STEPS:

        Step 1: Reshape from (B, W, N, D) to (B*N, W, D) for per-entity
                temporal processing, exactly like the original TAM does:
            x_node = x_node.permute(0, 2, 1, 3).reshape(-1, W, D_node)
            x_trace = x_trace.permute(0, 2, 1, 3).reshape(-1, W, D_edge)
            x_log = x_log.permute(0, 2, 1, 3).reshape(-1, W, D_log)

        Step 2: Run each through its Mamba block:
            x_node = self.mamba_node(x_node)    # (B*N, W, D_node)
            x_trace = self.mamba_trace(x_trace)  # (B*E, W, D_edge)
            x_log = self.mamba_log(x_log)        # (B*N, W, D_log)

        Step 3: Cross-modal mixing (Option A):
            # Reshape node and log back to (B, N, W, D) then to (B*N*W, D)
            # Concat: (B*N*W, D_node + D_log)
            # Project through self.mix_node_log
            # Split back into node and log portions
            #
            # Don't forget to add a residual connection around the mixing!

        Step 4: Reshape back to (B, W, N/E, D):
            x_node = x_node.reshape(B, N, W, D_node).permute(0, 2, 1, 3)
            x_trace = x_trace.reshape(B, E, W, D_edge).permute(0, 2, 1, 3)
            x_log = x_log.reshape(B, N, W, D_log).permute(0, 2, 1, 3)

        Step 5: Return x_node, x_trace, x_log
        """
        B, W, N, D_node = x_node.shape
        E = x_trace.shape[2]
        D_edge = x_trace.shape[3]
        D_log = x_log.shape[3]

        # Reshape to (B*N, W, D) for per-entity temporal processing
        x_node = x_node.permute(0, 2, 1, 3).reshape(B * N, W, D_node)
        x_trace = x_trace.permute(0, 2, 1, 3).reshape(B * E, W, D_edge)
        x_log = x_log.permute(0, 2, 1, 3).reshape(B * N, W, D_log)

        # Per-modality Mamba (Mamba is inherently causal; mask param unused)
        x_node = self.mamba_node(x_node)    # (B*N, W, D_node)
        x_trace = self.mamba_trace(x_trace)  # (B*E, W, D_edge)
        x_log = self.mamba_log(x_log)        # (B*N, W, D_log)

        # Cross-modal mixing for node and log (Option A)
        x_node_log = torch.cat([x_node, x_log], dim=-1)      # (B*N, W, D_node+D_log)
        mixed = self.mix_node_log(x_node_log) + x_node_log    # residual
        x_node = mixed[:, :, :self.node_dim]
        x_log = mixed[:, :, self.node_dim:]

        # Reshape back to (B, W, N/E, D)
        x_node = x_node.reshape(B, N, W, D_node).permute(0, 2, 1, 3)
        x_trace = x_trace.reshape(B, E, W, D_edge).permute(0, 2, 1, 3)
        x_log = x_log.reshape(B, N, W, D_log).permute(0, 2, 1, 3)

        return x_node, x_trace, x_log


# =============================================================================
# NEW: Association Discrepancy Module
# =============================================================================

class AssociationDiscrepancy(nn.Module):
    """Computes an anomaly score based on discrepancy between a learned
    prior distribution over SAM attention patterns and the actual (series)
    attention pattern observed at inference time.

    Motivation (from SARAD / Anomaly Transformer):
      Reconstruction error alone misses anomalies that manifest as
      distributional shifts in inter-service *relationships* rather than
      raw feature values. If Service A normally sends 80% of traffic to
      Service B and 20% to Service C, a cascading failure might flip this
      ratio. The reconstruction error on A's metrics could remain low, but
      the SAM attention pattern (which reflects the learned importance of
      graph neighbors) would deviate from its normal distribution.

    Design:
      - "Series association": the GAT attention coefficients alpha from SAM,
        extracted during forward pass. These are data-driven and change with
        each input window.
      - "Prior association": a learnable distribution (parameterized as
        log-probabilities over neighbor edges) that captures the expected
        normal attention pattern. Learned during training on normal data.
      - Discrepancy score: KL(prior || series) — high when the observed
        attention deviates from what the model learned to expect.

    Anchoring on SAM (not TAM/Mamba):
      SAM attention weights reflect inter-service spatial relationships.
      These are more semantically interpretable for catching cascading
      failures and dependency-structure anomalies than temporal patterns.

    Args:
        num_edges:   number of edges in the graph (for the prior distribution)
        num_heads:   number of GAT attention heads in SAM's node2node layer
    """
    def __init__(self, num_edges, num_heads):
        super().__init__()
        self.num_edges = num_edges
        self.num_heads = num_heads

        # TODO: Define the learnable prior distribution.
        #
        # The prior needs to produce a probability distribution over edges
        # for each head, to compare against the GAT alpha values.
        #
        # Option 1 (simple): One learnable logit vector per head
        #   self.prior_logits = nn.Parameter(torch.zeros(num_heads, num_edges))
        #   In forward: prior = softmax(self.prior_logits, dim=-1)
        #
        # Option 2 (richer): A small MLP that takes a global summary of
        #   the graph features and outputs the prior. This lets the prior
        #   adapt to the general "regime" of the system.
        #
        # Start with Option 1 — it's simpler and sufficient to validate
        # that association discrepancy adds value over reconstruction error.

        self.prior_logits = nn.Parameter(torch.zeros(num_heads, num_edges))

    def forward(self, node_attn_weights, edge_attn_weights=None):
        """
        Args:
            node_attn_weights: (num_edges_batched, num_heads) — alpha values
                               from SAM's node2node GATv2Conv layer.
                               Each row is a softmax-normalized attention coeff.
            edge_attn_weights: optional, same for the edge2node layer.

        Returns:
            discrepancy_score: scalar (or per-node) anomaly score based on
                               KL divergence between prior and series association.

        IMPLEMENTATION STEPS:

        Step 1: Reshape the attention weights.
            # The raw alpha from GATv2Conv is (total_edges_in_batch, num_heads).
            # "total_edges_in_batch" = B * W * edges_per_graph because the
            # batched sparse graph stacks all windows and batch items.
            #
            # To compute a meaningful per-graph-per-head distribution, you need
            # to reshape/scatter these back to per-graph groups.
            #
            # SIMPLIFICATION for first pass: average over the batch and window
            # dimensions to get a single (edges_per_graph, num_heads) matrix
            # representing the "typical" attention pattern for this batch.
            #
            # edges_per_graph = self.num_edges
            # total = node_attn_weights.shape[0]
            # num_copies = total // self.num_edges
            # alpha = node_attn_weights.reshape(num_copies, self.num_edges, self.num_heads)
            # alpha_mean = alpha.mean(dim=0)  # (num_edges, num_heads)

        Step 2: Compute the prior distribution.
            # prior = F.softmax(self.prior_logits, dim=-1)  # (num_heads, num_edges)
            # Transpose to match alpha_mean: prior.T → (num_edges, num_heads)

        Step 3: Compute KL divergence.
            # series_dist = alpha_mean (already softmax-normalized by GAT)
            # prior_dist = prior.T
            #
            # KL(prior || series) per head:
            #   kl = (prior_dist * (prior_dist.log() - series_dist.log())).sum(dim=0)
            #   → shape (num_heads,)
            #
            # Average over heads for a scalar discrepancy score:
            #   discrepancy = kl.mean()
            #
            # IMPORTANT: Add epsilon (1e-8) before log to avoid log(0).
            # Also clamp alpha values: series_dist = alpha_mean.clamp(min=1e-8)

        Step 4: Return discrepancy_score.
            # During training, this will be added to the loss.
            # During evaluation, it supplements the reconstruction-error-based
            # anomaly score.
        """
        # Step 1: Average over batch/window copies to get per-graph attention
        total = node_attn_weights.shape[0]
        num_copies = total // self.num_edges
        alpha = node_attn_weights[:num_copies * self.num_edges].reshape(
            num_copies, self.num_edges, self.num_heads)
        alpha_mean = alpha.mean(dim=0)                         # (num_edges, num_heads)

        # Step 2: Prior distribution (softmax over edges, per head)
        prior = F.softmax(self.prior_logits, dim=-1)           # (num_heads, num_edges)
        prior_T = prior.T                                       # (num_edges, num_heads)

        # Step 3: KL(prior || series)
        series = alpha_mean.clamp(min=1e-8)
        prior_clamped = prior_T.clamp(min=1e-8)
        kl = (prior_clamped * (prior_clamped.log() - series.log())).sum(dim=0)  # (num_heads,)
        return kl.mean()


# =============================================================================
# NEW: Encoder — uses MambaTemporalModule instead of Temporal_Attention
# =============================================================================

class Encoder(nn.Module):
    """Encoder with SAM + MambaTemporalModule + FFN per layer.

    Changes from MSTGAD Encoder:
      - Temporal_Attention → MambaTemporalModule
      - SAM can optionally return attention weights for association discrepancy

    Workflow per layer (same logical structure as MSTGAD):
      EI1 = Norm(EI0 + SAM(EI0))
      EI2 = Norm(EI1 + MambaTemp(EI1))     ← was TAM
      Z   = Norm(EI2 + FFN(EI2))
    """
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding,
                 node_heads, log_heads, edge_heads, n2e_heads, e2n_heads,
                 dropout, batch_size, window_size, num_layer, trace2pod,
                 d_state=16, d_conv=4, expand=2):
        super(Encoder, self).__init__()
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = \
            adj2adj(graph, batch_size, window_size, edge_embedding)
        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size

        self.spatial_attention = nn.ModuleList([
            Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                              heads_n2e=n2e_heads, heads_e2n=e2n_heads,
                              dropout=dropout, batch_size=batch_size,
                              window_size=window_size)
            for _ in range(self.L)
        ])
        self.sa_add = nn.ModuleList([
            AddALL(node_embedding, edge_embedding, log_embedding, dropout)
            for _ in range(self.L)
        ])

        # NEW: MambaTemporalModule replaces Temporal_Attention
        self.temporal_mamba = nn.ModuleList([
            MambaTemporalModule(
                node_embedding, edge_embedding, log_embedding, trace2pod,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, window_size=window_size, batch_size=batch_size)
            for _ in range(self.L)
        ])
        self.ta_add = nn.ModuleList([
            AddALL(node_embedding, edge_embedding, log_embedding, dropout)
            for _ in range(self.L)
        ])
        self.ffn = nn.ModuleList([
            FFN(node_embedding, edge_embedding, log_embedding, dropout)
            for _ in range(self.L)
        ])

    def forward(self, e_node, e_edge, e_log, return_attention_weights=False):
        """
        Args:
            e_node: (B, W, N, D_node) — embedded node features
            e_edge: (B, W, N, N, D_edge) — embedded edge features (dense)
            e_log:  (B, W, N, D_log)  — embedded log features
            return_attention_weights: if True, collect SAM attention weights
                                      from each layer for association discrepancy

        Returns:
            e_node, e_edge, e_log — encoder output Z_t
            If return_attention_weights: also returns list of (node_alpha, edge_alpha)
                                         per layer
        """
        # Mask edge features to sparse representation (same as MSTGAD)
        e_edge = torch.masked_select(e_edge, self.node_efea.bool()) \
            .reshape(e_edge.shape[0], e_edge.shape[1], -1, e_edge.shape[-1])

        all_attn_weights = []

        for i in range(self.L):
            # SAM
            if return_attention_weights:
                sam_out = self.spatial_attention[i](
                    e_node, e_edge, e_log, self.node_adj, self.edge_adj,
                    self.edge_efea, return_attention_weights=True)
                e_node, e_edge, e_log = self.sa_add[i](
                    e_node, e_edge, e_log, sam_out[0], sam_out[1], sam_out[2])
                all_attn_weights.append((sam_out[3], sam_out[4]))
            else:
                e_node, e_edge, e_log = self.sa_add[i](
                    e_node, e_edge, e_log,
                    *self.spatial_attention[i](
                        e_node, e_edge, e_log,
                        self.node_adj, self.edge_adj, self.edge_efea))

            # Mamba Temporal (replaces TAM)
            e_node, e_edge, e_log = self.ta_add[i](
                e_node, e_edge, e_log,
                *self.temporal_mamba[i](e_node, e_edge, e_log))

            # FFN
            e_node, e_edge, e_log = self.ffn[i](e_node, e_edge, e_log)

        if return_attention_weights:
            return e_node, e_edge, e_log, all_attn_weights
        return e_node, e_edge, e_log


# =============================================================================
# NEW: Decoder — uses MambaTemporalModule (inherently causal) instead of TAM
# =============================================================================

class Decoder(nn.Module):
    """Decoder with SAM + MambaTemporalModule (causal) + CAM + FFN per layer.

    Changes from MSTGAD Decoder:
      - Temporal_Attention → MambaTemporalModule
      - The mask=True parameter is still passed but Mamba is inherently causal,
        so no explicit masking is needed.

    Workflow per layer:
      DI1 = Norm(DI0 + SAM(DI0))
      DI2 = Norm(DI1 + MambaTemp(DI1))     ← was Mask(TAM(DI1))
      DI3 = Norm(DI2 + CAM(DI2, Z_t))
      O_t = Norm(DI3 + FFN(DI3))
    """
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding,
                 node_heads, log_heads, edge_heads, n2e_heads, e2n_heads,
                 dropout, batch_size, window_size, num_layer, trace2pod,
                 d_state=16, d_conv=4, expand=2):
        super(Decoder, self).__init__()
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = \
            adj2adj(graph, batch_size, window_size, edge_embedding)
        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size

        self.spatial_attention = nn.ModuleList([
            Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                              heads_n2e=n2e_heads, heads_e2n=e2n_heads,
                              dropout=dropout, batch_size=batch_size,
                              window_size=window_size)
            for _ in range(self.L)
        ])
        self.sa_add = nn.ModuleList([
            AddALL(node_embedding, edge_embedding, log_embedding, dropout)
            for _ in range(self.L)
        ])

        # NEW: MambaTemporalModule replaces Temporal_Attention
        self.temporal_mamba = nn.ModuleList([
            MambaTemporalModule(
                node_embedding, edge_embedding, log_embedding, trace2pod,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, window_size=window_size, batch_size=batch_size)
            for _ in range(self.L)
        ])
        self.ta_add = nn.ModuleList([
            AddALL(node_embedding, edge_embedding, log_embedding, dropout)
            for _ in range(self.L)
        ])

        # CAM unchanged
        self.cross_attention = nn.ModuleList([
            Encoder_Decoder_Attention(
                node_embedding, edge_embedding, log_embedding,
                heads_node=node_heads, heads_edge=edge_heads,
                heads_log=log_heads, dropout=dropout)
            for _ in range(self.L)
        ])
        self.ca_add = nn.ModuleList([
            AddALL(node_embedding, edge_embedding, log_embedding, dropout)
            for _ in range(self.L)
        ])
        self.ffn = nn.ModuleList([
            FFN(node_embedding, edge_embedding, log_embedding, dropout)
            for _ in range(self.L)
        ])

    def forward(self, d_node, d_edge, d_log, z_node, z_edge, z_log):
        d_edge = torch.masked_select(d_edge, self.node_efea.bool()) \
            .reshape(d_edge.shape[0], d_edge.shape[1], -1, d_edge.shape[-1])

        for i in range(self.L):
            # SAM
            d_node, d_edge, d_log = self.sa_add[i](
                d_node, d_edge, d_log,
                *self.spatial_attention[i](
                    d_node, d_edge, d_log,
                    self.node_adj, self.edge_adj, self.edge_efea))

            # MambaTemporalModule (inherently causal — mask param is vestigial)
            d_node, d_edge, d_log = self.ta_add[i](
                d_node, d_edge, d_log,
                *self.temporal_mamba[i](d_node, d_edge, d_log, mask=True))

            # CAM
            d_node, d_edge, d_log = self.ca_add[i](
                d_node, d_edge, d_log,
                *self.cross_attention[i](
                    d_node, d_edge, d_log, z_node, z_edge, z_log))

            # FFN
            d_node, d_edge, d_log = self.ffn[i](d_node, d_edge, d_log)

        return d_node, d_edge, d_log
