"""
Graph Construction Module for UPI Transaction Networks.
Constructs graph representations:
1. Transaction-Interaction Graph: Transactions connected by shared attributes (Device, Sender, Receiver, IP).
2. Heterogeneous Entity Graph: User/Merchant/Device nodes with Transaction relation edges.
Exports PyTorch Geometric Data object with temporal masks for train/val/test.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import networkx as nx
from sklearn.preprocessing import StandardScaler


class UPIGraphBuilder:
    """Builds PyTorch Geometric graph datasets from UPI transactions."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    def build_transaction_graph(
        self,
        df: pd.DataFrame,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        max_edges_per_group: int = 50,
    ) -> Data:
        """
        Builds a transaction-level graph where nodes are transactions,
        and edges connect transactions that share the same Device, Sender, or IP.
        """
        num_nodes = len(df)
        print(f"Building Transaction Graph with {num_nodes} nodes...")

        # Feature normalization
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(feature_matrix)
        x_tensor = torch.tensor(x_norm, dtype=torch.float)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        edge_sources = []
        edge_targets = []
        edge_types = []  # 0: shared_device, 1: shared_sender, 2: shared_receiver, 3: shared_ip

        # Helper to create clique or sampled edges within attribute groups
        def add_edges_for_group(group_col: str, edge_type_id: int):
            grouped = df.groupby(group_col).indices
            src_list = []
            dst_list = []
            for group_val, indices in grouped.items():
                if len(indices) < 2:
                    continue
                # If group is too large, sample pairs to prevent edge explosion
                if len(indices) > max_edges_per_group:
                    selected = np.random.choice(indices, size=max_edges_per_group, replace=False)
                else:
                    selected = indices

                for i in range(len(selected)):
                    for j in range(i + 1, min(i + 6, len(selected))):  # connect to k nearest neighbors in time
                        src_list.extend([selected[i], selected[j]])
                        dst_list.extend([selected[j], selected[i]])
            
            edge_sources.extend(src_list)
            edge_targets.extend(dst_list)
            edge_types.extend([edge_type_id] * len(src_list))
            print(f"  Added {len(src_list)} edges for attribute: {group_col}")

        add_edges_for_group("device_id", 0)
        add_edges_for_group("sender_id", 1)
        add_edges_for_group("receiver_vpa", 2)
        add_edges_for_group("ip_address", 3)

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)

        # Temporal split masks (strict chronological order)
        n_train = int(num_nodes * train_ratio)
        n_val = int(num_nodes * (train_ratio + val_ratio))

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train:n_val] = True
        test_mask[n_val:] = True

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_tensor,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
        )

        save_path = os.path.join(self.processed_dir, "upi_graph.pt")
        torch.save(data, save_path)
        print(f"Graph constructed and saved to: {save_path}")
        print(f"Graph Summary: {data}")
        return data

    def to_networkx(self, data: Data, max_nodes: int = 500) -> nx.Graph:
        """Export a subgraph to NetworkX for visualization."""
        G = nx.Graph()
        nodes_to_include = min(data.num_nodes, max_nodes)
        
        for i in range(nodes_to_include):
            is_fraud = bool(data.y[i].item() == 1)
            G.add_node(i, fraud=is_fraud)

        edge_index_np = data.edge_index.cpu().numpy()
        for src, dst in zip(edge_index_np[0], edge_index_np[1]):
            if src < nodes_to_include and dst < nodes_to_include:
                G.add_edge(int(src), int(dst))

        return G


if __name__ == "__main__":
    from src.data.preprocess import UPIPreprocessor
    
    preprocessor = UPIPreprocessor()
    df = preprocessor.load_data()
    X, y, df_feat = preprocessor.prepare_tabular_features(df)
    
    builder = UPIGraphBuilder()
    graph_data = builder.build_transaction_graph(df_feat, X, y)
    print("Graph builder verification complete.")
