"""
Configuration and hyperparameters for GNN-Based UPI Fraud Detection.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data processing configuration."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    dataset_name: str = "upi_transactions"

    # Graph construction
    node_types: List[str] = field(
        default_factory=lambda: ["user", "merchant", "device", "bank"]
    )
    edge_types: List[str] = field(
        default_factory=lambda: [
            "transacts_with",
            "uses_device",
            "has_account_at",
        ]
    )

    # Train/val/test split (temporal)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    """GNN model configuration."""
    model_type: str = "graphsage"  # "graphsage", "gat", "gcn"

    # Architecture
    in_channels: int = 64
    hidden_channels: int = 128
    out_channels: int = 2  # binary: fraud / not fraud
    num_layers: int = 2
    dropout: float = 0.3

    # GAT-specific
    num_heads: int = 4

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    epochs: int = 200
    batch_size: int = 512
    patience: int = 20  # early stopping

    # Neighbor sampling
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5])

    # Class imbalance
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    fraud_class_weight: float = 10.0

    # Checkpointing
    checkpoint_dir: str = "models"
    best_model_path: str = "models/best_model.pt"


@dataclass
class InferenceConfig:
    """Inference and API configuration."""
    model_path: str = "models/best_model.pt"
    fraud_threshold: float = 0.7
    max_latency_ms: float = 200.0

    # Rule-based layer
    rule_weight: float = 0.3  # final = (1 - rule_weight) * gnn + rule_weight * rules
    velocity_window_minutes: int = 5
    velocity_max_transactions: int = 5
    amount_std_threshold: float = 3.0


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    page_title: str = "UPI Fraud Detection — GNN Dashboard"
    refresh_interval_seconds: int = 2
    max_display_transactions: int = 100


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # Reproducibility
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"


# Global config instance
config = Config()
