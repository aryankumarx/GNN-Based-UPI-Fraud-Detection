import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "GraphSAGE_GNN" in data["models"]

def test_predict_endpoint_valid():
    payload = {
        "transaction_id": "TXN_TEST_9901",
        "timestamp": "2026-09-15 14:30:00",
        "sender_vpa": "user_00100@oksbi",
        "sender_bank": "SBI",
        "receiver_vpa": "merchant_99@paytm",
        "receiver_bank": "HDFC",
        "amount": 2500.0,
        "transaction_type": "P2P",
        "category": "Shopping",
        "device_id": "DEV_USER_001",
        "ip_address": "192.168.1.1",
        "location": "Delhi"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "gnn_score" in data
    assert "hybrid_risk_score" in data
    assert "decision" in data
    assert data["decision"] in ["APPROVE", "FLAG", "BLOCK", "REJECT", "REVIEW"]
