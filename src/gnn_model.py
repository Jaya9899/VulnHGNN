"""
gnn_model.py
============
GNN model for multi-label vulnerability detection on LLVM IR embeddings.

Architecture:
  VulnGNN:
    - Input projection  : 130 → 256
    - 3x Residual blocks: 256 → 256 (Linear + BN + ReLU + Dropout)
    - Attention pooling : learned weighted combination
    - Output head       : 256 → 128 → 6 (sigmoid)

Why this architecture:
  - Your graphs have no explicit successor edges in JSON, so we treat
    each graph's combined embedding as the node and build a deep MLP-GNN
    that mimics message passing via residual blocks.
  - Attention pooling lets the model focus on the most discriminative
    embedding dimensions per class.
  - Sigmoid output (not softmax) for true multi-label prediction.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 6


class ResidualBlock(nn.Module):
    """Linear residual block with BatchNorm and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))  # residual connection


class AttentionPooling(nn.Module):
    """
    Learned attention over embedding dimensions.
    Weighted combination → more expressive than mean pooling.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        weights = self.attn(x)
        return x * weights  # (B, dim)


class VulnGNN(nn.Module):
    """
    Multi-label vulnerability classifier.

    Input  : graph embedding vector (130-dim from feature_embeddings.py)
    Output : 6-dim logits (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        input_dim:   int   = 130,
        hidden_dim:  int   = 256,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = 0.3,
        n_layers:    int   = 3,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])

        # Attention pooling
        self.attn_pool = AttentionPooling(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.attn_pool(x)
        return self.classifier(x)  # raw logits


def build_model(input_dim: int = 130, dropout: float = 0.3) -> VulnGNN:
    return VulnGNN(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        n_layers=3,
    )