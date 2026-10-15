import pytest
import pandas as pd
import numpy as np

def test_preprocessing_synthetic_sanity():
    # Test sanity checks on UPI transaction payload
    sample_txn = {
        "transaction_id": "TXN_TEST_001",
        "amount": 2500.0,
        "sender_id": "user_001",
        "receiver_id": "merchant_001",
        "device_id": "DEV_001",
        "ip_address": "192.168.1.10"
    }
    
    assert sample_txn["amount"] > 0
    assert sample_txn["sender_id"] != sample_txn["receiver_id"]
    assert sample_txn["transaction_id"].startswith("TXN_")

def test_feature_scaling():
    amounts = np.array([100.0, 5000.0, 85000.0])
    log_amounts = np.log1p(amounts)
    
    assert len(log_amounts) == 3
    assert log_amounts[0] < log_amounts[1] < log_amounts[2]
