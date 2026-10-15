import pytest
from src.inference.rules import UPIRuleEngine

def test_rule_engine_mule_device():
    engine = UPIRuleEngine()
    
    # Mule device transaction
    mule_txn = {
        "amount": 85000.0,
        "device_id": "DEV_MULE_002",
        "ip_address": "10.0.99.4",
        "timestamp": "2026-09-15 03:30:00"
    }
    
    result = engine.evaluate_transaction(mule_txn)
    score = result["rule_score"]
    reasons = result["triggered_rules"]
    assert score >= 0.80
    assert len(reasons) > 0
    assert result["has_critical_rule"] is True

def test_rule_engine_clean_txn():
    engine = UPIRuleEngine()
    clean_txn = {
        "amount": 450.0,
        "device_id": "DEV_CLEAN_999",
        "ip_address": "192.168.1.5",
        "timestamp": "2026-09-15 14:30:00"
    }
    
    result = engine.evaluate_transaction(clean_txn)
    score = result["rule_score"]
    reasons = result["triggered_rules"]
    assert score < 0.20
    assert len(reasons) == 0
    assert result["has_critical_rule"] is False
