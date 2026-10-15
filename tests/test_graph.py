import pytest
import torch

def test_edge_index_format():
    # PyG edge_index format test: [2, num_edges]
    dummy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    assert dummy_edge_index.dim() == 2
    assert dummy_edge_index.size(0) == 2
    assert dummy_edge_index.size(1) == 3

def test_node_feature_dimensions():
    # 16-dimensional node feature tensor
    num_nodes = 100
    num_features = 16
    x = torch.randn(num_nodes, num_features)
    assert x.shape == (100, 16)
