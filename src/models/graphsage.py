"""
GraphSAGE (Sample and Aggregate) Architecture for Fraud Detection.
Uses multi-layer neighborhood aggregation to learn structural and relational representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm


class FraudGraphSAGE(nn.Module):
    """
    Inductive GraphSAGE model for node-level fraud classification.
    Supports 2 to 3 message passing layers with residual skip connections and batch normalization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="mean"))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Penultimate layer
        if num_layers >= 2:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Classification Head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass generating node embeddings and logits."""
        for i in range(self.num_layers):
            x_res = x if x.shape[-1] == self.convs[i].out_channels else None
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x_res is not None:
                x = x + x_res  # Residual skip connection

        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract intermediate GNN node embeddings (for visualization/clustering)."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        return x
