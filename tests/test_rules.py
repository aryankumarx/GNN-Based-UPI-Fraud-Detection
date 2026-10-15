import pytest
from src.inference.rules import RuleEngine

def test_rule_engine_mule_device():
    engine = RuleEngine()
    
    # Mule device transaction
    mule_txn = {
        "amount": 85000.0,
        "device_id": "DEV_MULE_002",
        "ip_address": "10.0.99.4",
        "timestamp": "2026-09-15 03:30:00"
    }
    
    score, reasons = engine.evaluate(mule_txn)
    assert score >= 0.80
    assert len(reasons) > 0
    assert any("DEV_MULE_002" in r for r in reasons)

def test_rule_engine_clean_txn():
    engine = RuleEngine()
    clean_txn = {
        "amount": 450.0,
        "device_id": "DEV_CLEAN_999",
        "ip_address": "192.168.1.5",
        "timestamp": "2026-09-15 14:30:00"
    }
    
    score, reasons = engine.evaluate(clean_txn)
    assert score < 0.20
    assert len(reasons) == 0
