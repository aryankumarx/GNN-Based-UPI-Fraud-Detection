"""
Graph Attention Network (GAT) Architecture for Fraud Detection.
Uses multi-head self-attention over neighboring transactions/entities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


class FraudGAT(nn.Module):
    """
    Graph Attention Network (GAT) with multi-head attention.
    Learns dynamic edge attention weights indicating which neighboring transactions are suspicious.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.heads = heads
        self.dropout = dropout

        # First GAT Conv Layer
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.bn1 = BatchNorm(hidden_channels * heads)

        # Second GAT Conv Layer
        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.bn2 = BatchNorm(hidden_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        if return_attention_weights:
            x, att1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = self.bn1(x)
            x = F.elu(x)
            x, att2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = self.bn2(x)
            x = F.elu(x)
            logits = self.classifier(x)
            return logits, (att1, att2)
        else:
            x = self.gat1(x, edge_index)
            x = self.bn1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gat2(x, edge_index)
            x = self.bn2(x)
            x = F.elu(x)
            logits = self.classifier(x)
            return logits
