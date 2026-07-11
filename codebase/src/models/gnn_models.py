"""gnn_models.py — GNN architectures for EGFR inhibitor classification.

Implements four Graph Neural Network models:
    - GCNModel       : Graph Convolutional Network (Kipf & Welling, 2017)
    - GATModel       : Graph Attention Network v2 (Brody et al., 2022)
    - GINModel       : Graph Isomorphism Network + Edge features (Hu et al., 2020)
    - AttentiveFPModel: Attentive FP (Xiong et al., 2020)

All models:
    - Accept node features of dimension 29 and edge features of dimension 12.
    - Output a single raw logit (no sigmoid) for use with BCEWithLogitsLoss.
    - Run on CPU only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    AttentiveFP,
    BatchNorm,
    GATv2Conv,
    GCNConv,
    GINEConv,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODE_DIM: int = 29
EDGE_DIM: int = 12


# ---------------------------------------------------------------------------
# Utility: concatenated mean + max readout
# ---------------------------------------------------------------------------

def _readout(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Concatenate global mean and max pooling over node embeddings."""
    return torch.cat([global_mean_pool(x, batch),
                      global_max_pool(x, batch)], dim=-1)


# ---------------------------------------------------------------------------
# GCNModel
# ---------------------------------------------------------------------------

class GCNModel(nn.Module):
    """Graph Convolutional Network for binary molecular classification.

    Uses degree-normalised spectral convolution (GCNConv) with BatchNorm
    and dropout, followed by concatenated mean+max global pooling and a
    linear classifier head.

    Args:
        in_channels: Node feature dimensionality (default 29).
        edge_dim: Edge feature dimensionality (unused in GCN; kept for API
            consistency).
        hidden_channels: Hidden embedding size per layer.
        num_layers: Number of GCNConv layers.
        dropout: Dropout probability applied after each layer.
        num_classes: Output logit dimensionality (default 1).
    """

    def __init__(
        self,
        in_channels: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_ch, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def reset_parameters(self) -> None:
        """Re-initialise all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = _readout(x, batch)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# GATModel
# ---------------------------------------------------------------------------

class GATModel(nn.Module):
    """Graph Attention Network v2 for binary molecular classification.

    Uses GATv2Conv with 4 attention heads and edge features. Intermediate
    layers concatenate heads; final layer averages them. Followed by
    concatenated mean+max pooling and a linear classifier.

    Args:
        in_channels: Node feature dimensionality (default 29).
        edge_dim: Edge feature dimensionality (default 12).
        hidden_channels: Per-head hidden size.
        num_layers: Number of GATv2Conv layers.
        dropout: Dropout probability.
        num_classes: Output logit dimensionality (default 1).
    """

    def __init__(
        self,
        in_channels: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 1,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels * heads
            concat = True if i < num_layers - 1 else False
            out_ch = hidden_channels
            self.convs.append(
                GATv2Conv(
                    in_ch, out_ch,
                    heads=heads,
                    concat=concat,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
            )
            bn_ch = out_ch * heads if concat else out_ch
            self.bns.append(BatchNorm(bn_ch))

        final_ch = hidden_channels  # concat=False on last layer
        self.classifier = nn.Linear(final_ch * 2, num_classes)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = _readout(x, batch)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# GINModel
# ---------------------------------------------------------------------------

class GINModel(nn.Module):
    """Graph Isomorphism Network with edge features (GINEConv).

    Achieves maximum expressiveness within the WL-equivalent message-passing
    framework via sum aggregation and a 2-layer MLP per convolution step.

    Args:
        in_channels: Node feature dimensionality (default 29).
        edge_dim: Edge feature dimensionality (default 12).
        hidden_channels: Hidden embedding size.
        num_layers: Number of GINEConv layers.
        dropout: Dropout probability.
        num_classes: Output logit dimensionality (default 1).
    """

    def __init__(
        self,
        in_channels: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(in_ch, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = _readout(x, batch)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# AttentiveFPModel
# ---------------------------------------------------------------------------

class AttentiveFPModel(nn.Module):
    """AttentiveFP graph neural network for molecular property prediction.

    Wraps PyTorch Geometric's built-in AttentiveFP implementation, which
    features atom-level attention and an iterative GRU-based molecular
    readout. Natively supports both node and edge features.

    Args:
        in_channels: Node feature dimensionality (default 29).
        edge_dim: Edge feature dimensionality (default 12).
        hidden_channels: Latent embedding size.
        num_layers: Number of message-passing layers.
        num_timesteps: Number of GRU readout timesteps.
        dropout: Dropout probability.
        num_classes: Output logit dimensionality (default 1).
    """

    def __init__(
        self,
        in_channels: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def reset_parameters(self) -> None:
        self.gnn.reset_parameters()
        self.classifier.reset_parameters()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.gnn(x, edge_index, edge_attr, batch)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# GraphSAGEModel
# ---------------------------------------------------------------------------

class GraphSAGEModel(nn.Module):
    """GraphSAGE for inductive molecular representation learning.

    Uses mean-aggregation neighbourhood sampling (Hamilton et al., 2017)
    with BatchNorm and dropout, followed by concatenated mean+max pooling.

    Args:
        in_channels: Node feature dimensionality (default 29).
        edge_dim: Edge feature dimensionality (unused in SAGE; kept for
            API consistency).
        hidden_channels: Hidden embedding size.
        num_layers: Number of SAGEConv layers.
        dropout: Dropout probability.
        num_classes: Output logit dimensionality (default 1).
    """

    def __init__(
        self,
        in_channels: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = _readout(x, batch)
        return self.classifier(x)
