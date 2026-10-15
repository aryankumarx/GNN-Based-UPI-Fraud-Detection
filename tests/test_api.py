import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gnn_model_loaded" in data

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_scanned" in data
    assert "fraud_detected" in data
    assert "avg_latency_ms" in data

def test_predict_endpoint_valid():
    payload = {
        "sender_id": "U1001",
        "receiver_id": "U9999",
        "amount": 2500.0,
        "device_id": "DEV_X1",
        "ip_address": "192.168.1.1",
        "location": "DELHI",
        "is_weekend": 0,
        "hour_of_day": 14
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "gnn_prob" in data
    assert "hybrid_score" in data
    assert "action" in data
    assert data["action"] in ["APPROVE", "FLAG", "BLOCK"]
