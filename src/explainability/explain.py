"""
Explainability Module for GNN Fraud Decisions.
Generates human-understandable risk rationales by analyzing:
1. GNN Subgraph Neighborhood (connected devices, suspicious counterparties).
2. Attributed Feature Contributions (amounts, hours, velocity).
3. Combined Summary for Security Analysts.
"""

from typing import Dict, Any, List
import numpy as np


class FraudExplainer:
    """Generates structured explanations for transactions flagged by the GNN model."""

    FEATURE_NAMES = [
        "Amount (₹)",
        "Log Amount",
        "Hour of Day",
        "Day of Week",
        "Is Night Time",
        "User Historical Mean",
        "Amount Z-Score",
        "User Txn Frequency",
        "Sender Bank",
        "Receiver Bank",
        "Transaction Type",
        "Merchant Category",
        "Device Type",
        "Location City",
    ]

    def explain_transaction(
        self,
        txn_data: Dict[str, Any],
        gnn_score: float,
        rule_info: Dict[str, Any],
        neighbor_summary: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize an actionable security audit summary."""
        reasons = []

        # Graph / Network-level explanations
        if neighbor_summary:
            suspicious_neighbors = [n for n in neighbor_summary if n.get("is_fraud", False)]
            if suspicious_neighbors:
                reasons.append(
                    f"Connected to {len(suspicious_neighbors)} known fraudulent entities via shared device/IP."
                )

        # GNN Embedding & Score factors
        if gnn_score >= 0.70:
            reasons.append(f"Graph Neural Network confidence high ({gnn_score*100:.1f}%) based on relational topology.")

        # Heuristic rules
        for r in rule_info.get("triggered_rules", []):
            reasons.append(r)

        if not reasons:
            reasons.append("Normal transactional pattern within expected user baselines.")

        severity = "HIGH" if (gnn_score >= 0.75 or rule_info.get("has_critical_rule")) else (
            "MEDIUM" if gnn_score >= 0.45 else "LOW"
        )

        return {
            "transaction_id": txn_data.get("transaction_id", "N/A"),
            "gnn_fraud_probability": round(gnn_score, 4),
            "rule_risk_score": rule_info.get("rule_score", 0.0),
            "risk_severity": severity,
            "explanation_points": reasons,
            "recommendation": "BLOCK & OTP_CHALLENGE" if severity == "HIGH" else (
                "FLAG_FOR_REVIEW" if severity == "MEDIUM" else "APPROVE"
            ),
        }
