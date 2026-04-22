"""
pyg_model.py: Graph Attention Network (GAT) for multi-label vulnerability detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops

NUM_CLASSES      = 5
INST_FEATURE_DIM = 22
EDGE_TYPES       = 3   # sequential=0, def_use=1, cfg=2


class VulnGAT(nn.Module):
    """
    Multi-label vulnerability classifier using Graph Attention Networks.

    Input  : PyG Data object with x (node features), edge_index, edge_attr
    Output : 6-dim logits (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        in_channels:  int   = INST_FEATURE_DIM,
        hidden:       int   = 128,
        out_channels: int   = 64,
        num_classes:  int   = NUM_CLASSES,
        heads:        int   = 4,
        dropout:      float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout

        # Edge type embedding — gives each edge type a learned vector
        # that gets added to node features during message passing
        self.edge_emb = nn.Embedding(EDGE_TYPES, 8)

        # Input projection (node features + edge type context)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # GAT layers
        # Layer 1: hidden → hidden (multi-head)
        self.gat1 = GATv2Conv(
            in_channels  = hidden,
            out_channels = hidden // heads,
            heads        = heads,
            dropout      = dropout,
            edge_dim     = 8,       # edge type embedding dim
            concat       = True,    # concat heads → hidden
        )
        self.bn1 = nn.BatchNorm1d(hidden)

        # Layer 2: hidden → hidden (multi-head)
        self.gat2 = GATv2Conv(
            in_channels  = hidden,
            out_channels = hidden // heads,
            heads        = heads,
            dropout      = dropout,
            edge_dim     = 8,
            concat       = True,
        )
        self.bn2 = nn.BatchNorm1d(hidden)

        # Layer 3: hidden → out_channels (fewer heads for compression)
        self.gat3 = GATv2Conv(
            in_channels  = hidden,
            out_channels = out_channels // 2,
            heads        = 2,
            dropout      = dropout,
            edge_dim     = 8,
            concat       = True,   # → out_channels
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        # After mean+max pool → out_channels * 2
        pool_dim = out_channels * 2

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x          : [num_nodes, 18]       node features
            edge_index : [2, num_edges]         edge connectivity
            edge_attr  : [num_edges]            edge type indices (0/1/2)
            batch      : [num_nodes]            graph assignment per node
        """
        # Get edge type embeddings
        e = self.edge_emb(edge_attr)  # [num_edges, 8]

        # Input projection
        x = self.input_proj(x)        # [num_nodes, hidden]

        # GAT Layer 1 + residual
        x1 = self.gat1(x, edge_index, edge_attr=e)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x  = x + x1   # residual (same dim)

        # GAT Layer 2 + residual
        x2 = self.gat2(x, edge_index, edge_attr=e)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x  = x + x2   # residual

        # GAT Layer 3 (compression, no residual — dim changes)
        x = self.gat3(x, edge_index, edge_attr=e)
        x = self.bn3(x)
        x = F.elu(x)

        # Global pooling: mean + max → richer graph representation
        x_mean = global_mean_pool(x, batch)  # [batch_size, out_channels]
        x_max  = global_max_pool(x, batch)   # [batch_size, out_channels]
        x      = torch.cat([x_mean, x_max], dim=-1)  # [batch_size, out_channels*2]

        return self.classifier(x)  # [batch_size, num_classes]  raw logits


def build_pyg_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3, input_dim: int = INST_FEATURE_DIM) -> VulnGAT:
    return VulnGAT(
        in_channels  = input_dim,
        hidden       = 128,
        out_channels = 64,
        num_classes  = num_classes,
        heads        = 4,
        dropout      = dropout,
    )