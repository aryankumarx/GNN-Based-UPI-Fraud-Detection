"""
Baseline Models for Benchmark Comparison:
1. Logistic Regression
2. MLP (Multi-Layer Perceptron)
3. XGBoost
4. LightGBM
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from src.training.evaluate import evaluate_predictions, print_metrics_table


class TabularMLP(nn.Module):
    """Deep Neural Network baseline operating purely on tabular node features (no graph)."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    """Train XGBoost or GradientBoosting with class weighting for class imbalance."""
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    
    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model_title = "XGBoost Baseline"
    else:
        # Fallback to Scikit-Learn HistGradientBoosting
        class_weight_dict = {0: 1.0, 1: pos_weight}
        model = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            learning_rate=0.05,
            class_weight=class_weight_dict,
            random_state=42,
        )
        model_title = "GradientBoosting Baseline (HistGB)"

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob)
    print_metrics_table(model_title, metrics)
    return model, metrics


def train_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 0.001,
) -> Tuple[TabularMLP, Dict[str, Any]]:
    """Train Tabular MLP baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(in_features=X_train.shape[1]).to(device)

    # Class weighting
    class_counts = np.bincount(y_train)
    weights = torch.tensor([1.0, class_counts[0] / max(1, class_counts[1])], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float).to(device)
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    metrics = evaluate_predictions(y_test, probs)
    print_metrics_table("Tabular MLP Baseline", metrics)
    return model, metrics
