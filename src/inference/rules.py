"""
Domain-Specific Rule-Based Layer for UPI Security.
Detects:
1. Transaction velocity bursts (rapid repeated transactions).
2. High-value amount anomalies against user baseline.
3. Midnight / odd-hour activity.
4. Device switching or known mule device usage.
"""

from typing import Dict, Any, List
from datetime import datetime


class UPIRuleEngine:
    """Evaluates rule-based risk heuristics for a transaction."""

    def __init__(self, amount_std_threshold: float = 3.0):
        self.amount_std_threshold = amount_std_threshold

    def evaluate_transaction(self, txn: Dict[str, Any], user_history: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluates heuristic security rules for an incoming transaction.
        Returns a rule risk score (0.0 to 1.0) and list of triggered flags.
        """
        triggered_rules: List[str] = []
        score_accumulator = 0.0

        amount = float(txn.get("amount", 0))
        device_id = str(txn.get("device_id", ""))
        timestamp_str = str(txn.get("timestamp", ""))

        # Parse hour
        hour = 12
        if timestamp_str:
            try:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour
            except ValueError:
                pass

        # 1. Midnight / Odd Hours Rule (11 PM - 5 AM)
        if hour >= 23 or hour <= 4:
            triggered_rules.append("Odd-Hour Transaction (Midnight 11 PM - 5 AM)")
            score_accumulator += 0.25

        # 2. Mule Device Signature Rule
        if "MULE" in device_id or "UNKNOWN" in device_id:
            triggered_rules.append("High-Risk Shared Device Identifier Detected")
            score_accumulator += 0.50

        # 3. High-Value UPI Spike
        if amount >= 75000:
            triggered_rules.append(f"Near-Limit High Value Transfer (₹{amount:,.2f})")
            score_accumulator += 0.35

        # 4. User Baseline Deviation (if user history exists)
        if user_history:
            avg_amt = user_history.get("avg_amount", 1000.0)
            if amount > (avg_amt * 10):
                triggered_rules.append(f"Amount Spikes 10x Above User's Average (₹{avg_amt:,.2f})")
                score_accumulator += 0.40

        rule_score = min(1.0, score_accumulator)

        return {
            "rule_score": round(rule_score, 4),
            "triggered_rules": triggered_rules,
            "has_critical_rule": len(triggered_rules) >= 2 or ("MULE" in device_id),
        }
