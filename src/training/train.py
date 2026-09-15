"""
Main Training and Benchmarking Pipeline for GNN UPI Fraud Detection.
Trains:
1. Baselines (XGBoost, Tabular MLP)
2. GraphSAGE
3. GAT
Outputs a comprehensive benchmark comparison table.
"""

import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader

from src.data.preprocess import UPIPreprocessor
from src.data.graph_builder import UPIGraphBuilder
from src.models.graphsage import FraudGraphSAGE
from src.models.gat import FraudGAT
from src.models.baselines import train_xgboost_baseline, train_mlp_baseline
from src.training.losses import FocalLoss
from src.training.evaluate import evaluate_predictions, print_metrics_table
from src.utils.config import config


def train_gnn_model(
    model: nn.Module,
    data,
    model_name: str = "GraphSAGE",
    epochs: int = 40,
    lr: float = 0.003,
    device: str = "cpu",
    save_path: str = "models/best_model.pt",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train GNN with Focal Loss and early stopping."""
    print(f"\n🚀 Training {model_name}...")
    model = model.to(device)
    data = data.to(device)

    # Focal loss handles rare fraud
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_f1 = 0.0
    best_metrics = {}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_val = data.y[data.val_mask].cpu().numpy()
            probs_val = probs[data.val_mask.cpu().numpy()]
            val_metrics = evaluate_predictions(y_val, probs_val)

        scheduler.step(val_metrics["f1_score"])

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {loss.item():.4f} | Val F1: {val_metrics['f1_score']*100:.2f}% | Val AUPRC: {val_metrics['auprc']:.4f}")

    train_time = time.time() - start_time
    print(f"✅ Finished training in {train_time:.2f}s (Best Val F1: {best_val_f1*100:.2f}%)")

    # Load best checkpoint for test evaluation
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_test = data.y[data.test_mask].cpu().numpy()
        probs_test = probs[data.test_mask.cpu().numpy()]
        test_metrics = evaluate_predictions(y_test, probs_test)

    test_metrics["training_time_sec"] = round(train_time, 2)
    print_metrics_table(f"{model_name} (Test Set)", test_metrics)
    return model, test_metrics


def run_full_pipeline():
    """Executes the full pipeline: Data -> Graph -> Baselines -> GNNs -> Comparison."""
    print("=" * 60)
    print("🛡️ GNN-Based UPI Transaction Fraud Detection Pipeline")
    print("=" * 60)

    # 1. Load / Preprocess Data
    preprocessor = UPIPreprocessor()
    df = preprocessor.load_data()
    X, y, df_feat = preprocessor.prepare_tabular_features(df)
    splits = preprocessor.temporal_split(X, y)

    # 2. Train Baselines
    print("\n[Step 1/3] Benchmarking Baseline Models (Tabular / No Graph)...")
    xgb_model, xgb_metrics = train_xgboost_baseline(
        splits["X_train"], splits["y_train"], splits["X_test"], splits["y_test"]
    )
    mlp_model, mlp_metrics = train_mlp_baseline(
        splits["X_train"], splits["y_train"], splits["X_test"], splits["y_test"], epochs=20
    )

    # 3. Build Graph
    print("\n[Step 2/3] Constructing Relational Transaction Graph...")
    builder = UPIGraphBuilder()
    graph_data = builder.build_transaction_graph(df_feat, X, y)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Train GraphSAGE
    print("\n[Step 3/3] Training Graph Neural Networks...")
    in_channels = graph_data.x.shape[1]
    sage_model = FraudGraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=2, num_layers=2)
    sage_model, sage_metrics = train_gnn_model(
        sage_model, graph_data, model_name="GraphSAGE", epochs=35, device=device, save_path="models/graphsage_best.pt"
    )

    # 5. Train GAT
    gat_model = FraudGAT(in_channels=in_channels, hidden_channels=32, heads=4, out_channels=2)
    gat_model, gat_metrics = train_gnn_model(
        gat_model, graph_data, model_name="GAT (Graph Attention Network)", epochs=35, device=device, save_path="models/gat_best.pt"
    )

    # 6. Final Benchmark Summary Table
    print("\n" + "=" * 70)
    print("🏆 FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUPRC':<10} | {'ROC-AUC':<10}")
    print("-" * 70)
    
    results = [
        ("XGBoost (Tabular)", xgb_metrics),
        ("MLP (Deep Neural)", mlp_metrics),
        ("GraphSAGE (GNN)", sage_metrics),
        ("GAT (Attention GNN)", gat_metrics),
    ]

    for name, m in results:
        print(f"{name:<25} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1_score']*100:>8.2f}% | {m['auprc']:>10.4f} | {m['roc_auc']:>10.4f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_full_pipeline()
