"""
MESTGAD.py — Mamba-Enhanced Spatial-Temporal Graph Anomaly Detection

Extensions over MSTGAD (MyModel):
  1. TAM → MambaTemporalModule (linear-time temporal encoding)
  2. Association discrepancy scoring on SAM attention weights
     (supplements reconstruction-error-based anomaly detection)

Loss function becomes:
  Loss = (1/epoch) * L1 + (1 - 1/epoch) * L2 + lambda_ad * L_ad

  where L_ad is the association discrepancy loss (KL divergence between
  learned prior and observed SAM attention distributions).

  During evaluation, the anomaly score is:
  score = reconstruction_error + lambda_ad * association_discrepancy
"""

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from src.MESTGAD_util import *


class MESTGADModel(nn.Module):
    """MESTGAD: Mamba-Enhanced Spatial-Temporal Graph Anomaly Detection.

    Architecture overview (cf. MSTGAD Fig. 1 and Fig. 2):

    ┌─────────────────────────────────────────────────────────┐
    │  Input Embedding (node, edge, log)                      │
    │      ↓                                                  │
    │  Encoder (L layers):                                    │
    │    SAM (graph attention, extracts attention weights)    │
    │    → MambaTemporalModule (replaces TAM)                 │
    │    → FFN                                                │
    │      ↓  Z_t                                             │
    │  Decoder (L layers):                                    │
    │    SAM → MambaTemp (causal) → CAM(Z_t) → FFN            │
    │      ↓  O_t                                             │
    │  Reconstruction: ||G_t - G'_t||^2                       │
    │  Classification: MLP → Softmax → P_t                    │
    │  Assoc Discrepancy: KL(prior || SAM_alpha)              │
    └─────────────────────────────────────────────────────────┘

    Args (via **args dict):
        --- inherited from MSTGAD ---
        label_weight:    weight for unknown-label reconstruction loss
        feature_node:    node embedding dimension
        feature_edge:    edge embedding dimension
        feature_log:     log embedding dimension
        num_heads_node, num_heads_log, num_heads_edge: attention heads
        num_heads_n2e, num_heads_e2n: GAT heads for SAM
        dropout:         dropout rate
        batch_size:      B
        window:          W (sliding window size)
        num_layer:       L (encoder/decoder layers)
        raw_node:        raw metric feature dimension
        raw_edge:        raw trace feature dimension
        log_len:         raw log feature dimension

        --- new for MESTGAD ---
        d_state:         Mamba SSM hidden state dimension (default 16)
        d_conv:          Mamba local conv kernel width (default 4)
        expand:          Mamba expansion factor (default 2)
        lambda_ad:       weight for association discrepancy loss (default 0.1)
    """
    def __init__(self, graph, **args):
        super(MESTGADModel, self).__init__()
        self.name = 'mestgad'
        self.graph = torch.tensor(graph).cuda()
        self.label_weight = args['label_weight']

        # Mamba hyperparameters (with defaults for backward compat)
        self.d_state = args.get('d_state', 16)
        self.d_conv = args.get('d_conv', 4)
        self.expand = args.get('expand', 2)
        self.lambda_ad = args.get('lambda_ad', 0.1)

        # --- Graph topology setup (unchanged from MSTGAD) ---
        adj = dense_to_sparse(self.graph)[0]
        trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=graph.shape[0]) \
            + torch.nn.functional.one_hot(adj[1], num_classes=graph.shape[0])
        trace2pod = trace2pod / trace2pod.sum(axis=0, keepdim=True)
        trace2pod = torch.where(
            torch.isnan(trace2pod), torch.full_like(trace2pod, 0), trace2pod)

        # --- Encoder & Decoder (now with Mamba temporal modules) ---
        encoder_decoder_kwargs = dict(
            graph=self.graph,
            node_embedding=args['feature_node'],
            edge_embedding=args['feature_edge'],
            log_embedding=args['feature_log'],
            node_heads=args['num_heads_node'],
            log_heads=args['num_heads_log'],
            edge_heads=args['num_heads_edge'],
            n2e_heads=args['num_heads_n2e'],
            e2n_heads=args['num_heads_e2n'],
            dropout=args['dropout'],
            batch_size=args['batch_size'],
            window_size=args['window'],
            num_layer=args['num_layer'],
            trace2pod=trace2pod,
            # New Mamba params
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        self.encoder = Encoder(**encoder_decoder_kwargs)
        self.decoder = Decoder(**encoder_decoder_kwargs)

        # --- Input embeddings (unchanged) ---
        self.node_emb = Embed(args['raw_node'], args['feature_node'], dim=4)
        self.log_emb = Embed(args['log_len'], args['feature_log'], dim=4)
        self.egde_emb = Embed(args['raw_edge'], args['feature_edge'], dim=5)

        # --- trace2pod mapping for reconstruction (unchanged) ---
        self.trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=self.graph.shape[0]) \
            + torch.nn.functional.one_hot(adj[1], num_classes=self.graph.shape[0])
        self.trace2pod = self.trace2pod / 2

        # --- Output projection heads (unchanged) ---
        self.dense_node = nn.Linear(args['feature_node'], args['raw_node'])
        self.dense_log = nn.Linear(args['feature_log'], args['log_len'])
        self.dense_edge = nn.Linear(args['feature_edge'], args['raw_edge'])

        # --- Classification head (unchanged) ---
        self.show = nn.Sequential(
            nn.Linear(
                args['raw_node'] + args['raw_edge'] + args['log_len'],
                (args['raw_node'] + args['raw_edge'] + args['log_len']) // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(
                (args['raw_node'] + args['raw_edge'] + args['log_len']) // 2,
                2))

        # --- NEW: Association Discrepancy module ---
        num_edges = adj.shape[1]
        self.assoc_discrepancy = AssociationDiscrepancy(
            num_edges=num_edges,
            num_heads=args['num_heads_n2e']
        )

    def forward(self, x, evaluate=False):
        """
        Args:
            x: dict with keys:
                'data_node':       (B, W, N, F_node_raw)
                'data_edge':       (B, W, N, N, F_edge_raw)
                'data_log':        (B, W, N, F_log_raw)
                'groundtruth_cls': (B, N, 2) or (B, N, 3) — one-hot labels
            evaluate: if True, return softmax predictions and labels

        Returns (training):
            rec_loss:    list of [normal_rec, abnormal_rec, unknown_rec] scalars
            cls_result:  (B*N, 2) classification logits
            cls_label:   (B*N, 2) one-hot ground truth
            ad_loss:     scalar association discrepancy loss (NEW)

        Returns (evaluation):
            cls_result:  (N, 2) softmax probabilities
            cls_label:   ground truth labels
            ad_score:    scalar association discrepancy score (NEW)
        """
        # --- Input embedding (unchanged) ---
        x_node, d_node = self.node_emb(x['data_node'])
        x_edge, d_edge = self.egde_emb(x['data_edge'])
        x_log, d_log = self.log_emb(x['data_log'])

        # --- Encoder ---
        z_node, z_edge, z_log, attn_weights_list = self.encoder(
            x_node, x_edge, x_log, return_attention_weights=True)

        # --- Decoder (unchanged logic) ---
        node, edge, log = self.decoder(d_node, d_edge, d_log,
                                       z_node, z_edge, z_log)

        # --- Reconstruction error computation (unchanged from MSTGAD) ---
        l_edge = torch.masked_select(
            x['data_edge'],
            self.graph.unsqueeze(-1).repeat(
                1, 1, x['data_edge'].shape[-1]).bool()
        ).reshape(x['data_edge'].shape[0], x['data_edge'].shape[1],
                  -1, x['data_edge'].shape[-1])

        rec_node = torch.square(self.dense_node(node) - x['data_node'])
        rec_edge1 = torch.square(self.dense_edge(edge) - l_edge)
        rec_log = torch.square(self.dense_log(log) - x['data_log'])

        rec_edge = torch.matmul(
            rec_edge1.permute(0, 1, 3, 2),
            self.trace2pod.float()
        ).permute(0, 1, 3, 2)
        rec = torch.concat([rec_node, rec_log, rec_edge], dim=-1)

        # --- Association discrepancy score ---
        ad_scores = []
        for (node_alpha, edge_alpha) in attn_weights_list:
            if node_alpha is not None:
                ad_scores.append(self.assoc_discrepancy(node_alpha, edge_alpha))
        ad_loss = torch.stack(ad_scores).mean() if ad_scores else torch.tensor(0.0, device=rec.device)

        if evaluate:
            rec = rec[:, -1].squeeze()
            cls_result = torch.softmax(self.show(rec), dim=-1)
            return cls_result, x['groundtruth_cls'], ad_loss
        else:
            cls_label = x['groundtruth_cls']

            rec = rec[:, -1].squeeze()
            cls_result = self.show(rec)
            cls_result = cls_result.reshape(-1, cls_result.shape[-1])
            cls_label = cls_label.reshape(-1, cls_label.shape[-1])

            if cls_label.shape[-1] == 3:
                mask = cls_label[:, -1]
                cls_result, cls_label = cls_result[mask == 0], cls_label[mask == 0]
                cls_label = cls_label[:, :cls_result.shape[-1]]

            # --- Reconstruction loss (unchanged) ---
            label_pod = torch.argmax(x['groundtruth_cls'], dim=-1)
            node_rec = torch.sum(rec, dim=-1)
            node_right = torch.where(
                label_pod == 0, node_rec,
                torch.zeros_like(node_rec).to(node_rec.device))
            node_wrong = torch.where(
                label_pod == 1,
                torch.pow(node_rec, torch.tensor(-1, device=node_rec.device)),
                torch.zeros_like(node_rec).to(node_rec.device))
            node_unkown = torch.where(
                label_pod == 2,
                self.label_weight * node_rec,
                torch.zeros_like(node_rec).to(node_rec.device))
            rec_loss = [node_right, node_wrong, node_unkown]

            param = label_pod.shape[0] * label_pod.shape[1]
            rec_loss = list(map(lambda x: x.sum() / param, rec_loss))

            # Return ad_loss as fourth element for the training loop to use
            return rec_loss, cls_result, cls_label, ad_loss


# =============================================================================
# Training loop modifications (reference — not a runnable function)
# =============================================================================

"""
TODO: Modify the training loop to incorporate ad_loss.

The original MSTGAD training loop (Algorithm 1) computes:
    L = (1/epoch) * L1 + (1 - 1/epoch) * L2

For MESTGAD, extend to:
    L = (1/epoch) * L1 + (1 - 1/epoch) * L2 + lambda_ad * L_ad

Where L_ad = ad_loss returned by the forward pass.

Pseudocode for the training step:

    rec_loss, cls_result, cls_label, ad_loss = model(batch_data)

    # L1: semi-supervised reconstruction loss (unchanged)
    L1 = eta * rec_loss[0].sum() + rec_loss[1].sum() + rec_loss[2].sum()
    # or however your current training code aggregates rec_loss

    # L2: classification loss (unchanged)
    L2 = F.cross_entropy(cls_result, cls_label, weight=class_weights)

    # L_ad: association discrepancy loss (NEW)
    L_ad = ad_loss

    # Combined loss
    epoch_frac = 1.0 / current_epoch
    loss = epoch_frac * L1 + (1 - epoch_frac) * L2 + model.lambda_ad * L_ad

    loss.backward()
    optimizer.step()


For evaluation, the forward pass now returns:
    cls_result, cls_label, ad_score = model(batch_data, evaluate=True)

You can combine the classification probability with the ad_score:
    anomaly_score = (1 - cls_result[:, 0]) + model.lambda_ad * ad_score

Or keep them separate and analyze both signals independently.
"""
