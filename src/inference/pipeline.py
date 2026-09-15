"""
Real-Time Hybrid Inference Pipeline.
Combines GNN graph score, domain rules, and explainability to output a unified risk decision.
"""

import os
from typing import Dict, Any
import numpy as np
import torch

from src.inference.rules import UPIRuleEngine
from src.explainability.explain import FraudExplainer
from src.models.graphsage import FraudGraphSAGE
from src.utils.config import config


class RealTimeFraudPipeline:
    """End-to-end real-time fraud scoring pipeline."""

    def __init__(
        self,
        model_path: str = "models/graphsage_best.pt",
        in_channels: int = 14,
        rule_weight: float = 0.30,
        fraud_threshold: float = 0.65,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FraudGraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=2)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained GNN model weights from {model_path}")
        else:
            print("Notice: Model weights not found yet, operating in fallback rule/untrained mode.")

        self.model.eval()
        self.rule_engine = UPIRuleEngine()
        self.explainer = FraudExplainer()
        self.rule_weight = rule_weight
        self.fraud_threshold = fraud_threshold

    def process_transaction(
        self,
        txn: Dict[str, Any],
        feature_vector: np.ndarray = None,
        neighbor_summary: list = None,
        user_history: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Score an incoming transaction and generate explainability audit."""
        # 1. Evaluate Rule Engine
        rule_res = self.rule_engine.evaluate_transaction(txn, user_history)
        rule_score = rule_res["rule_score"]

        # 2. Evaluate GNN Model
        if feature_vector is not None:
            with torch.no_grad():
                x_t = torch.tensor(feature_vector, dtype=torch.float).unsqueeze(0).to(self.device)
                # Self-loop edge for isolated new inference node
                dummy_edge = torch.zeros((2, 1), dtype=torch.long).to(self.device)
                logits = self.model(x_t, dummy_edge)
                gnn_score = float(torch.softmax(logits, dim=1)[0, 1].item())
        else:
            # Heuristic simulation if raw vector not supplied
            gnn_score = rule_score * 0.85

        # 3. Hybrid Combined Score
        hybrid_score = ((1.0 - self.rule_weight) * gnn_score) + (self.rule_weight * rule_score)
        is_fraud_decision = bool(hybrid_score >= self.fraud_threshold or rule_res["has_critical_rule"])

        # 4. Generate Explainability
        explanation = self.explainer.explain_transaction(
            txn, gnn_score=gnn_score, rule_info=rule_res, neighbor_summary=neighbor_summary
        )

        return {
            "transaction_id": txn.get("transaction_id", "TXN_000"),
            "sender_vpa": txn.get("sender_vpa", ""),
            "receiver_vpa": txn.get("receiver_vpa", ""),
            "amount": float(txn.get("amount", 0)),
            "gnn_score": round(gnn_score, 4),
            "rule_score": round(rule_score, 4),
            "hybrid_risk_score": round(hybrid_score, 4),
            "is_fraud": is_fraud_decision,
            "severity": explanation["risk_severity"],
            "decision": explanation["recommendation"],
            "explanations": explanation["explanation_points"],
        }
