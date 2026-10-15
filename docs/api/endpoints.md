# UPI Fraud Detection Engine - REST API Documentation

The UPI Fraud Detection backend is built on **FastAPI** and provides high-throughput, sub-10ms endpoints for real-time transaction inference, health monitoring, and system metrics.

---

## 1. Predict Transaction (`POST /predict`)

Evaluates a single UPI transaction through the **70% GraphSAGE GNN + 30% Heuristic Rule Engine** hybrid classifier.

### Request Headers
```http
Content-Type: application/json
```

### Request Payload (`TransactionInput`)
```json
{
  "sender_id": "U1001",
  "receiver_id": "U9999",
  "amount": 45000.0,
  "device_id": "DEV_ANDROID_84",
  "ip_address": "103.21.244.0",
  "location": "DELHI",
  "is_weekend": 0,
  "hour_of_day": 3
}
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `sender_id` | `string` | Yes | Sender User ID / VPA |
| `receiver_id` | `string` | Yes | Recipient User ID / VPA |
| `amount` | `float` | Yes | Transaction amount in INR (> 0.0) |
| `device_id` | `string` | No | Hardware device fingerprint ID |
| `ip_address` | `string` | No | Originating client IP address |
| `location` | `string` | No | Originating city / state |
| `is_weekend` | `integer` | No | `1` if Saturday/Sunday, `0` otherwise |
| `hour_of_day` | `integer` | No | Hour in 24-hr format (`0-23`) |

### Response (`200 OK`)
```json
{
  "is_fraud": true,
  "hybrid_score": 0.8124,
  "gnn_prob": 0.8410,
  "rule_score": 0.7450,
  "action": "BLOCK",
  "rule_reasons": [
    "High velocity burst detected",
    "Unusual nocturnal activity (01:00 - 05:00)"
  ],
  "decision_threshold": 0.35,
  "latency_ms": 6.84
}
```

---

## 2. Real-Time Telemetry Stats (`GET /stats`)

Returns aggregated fraud scanning counters, block rates, and rolling inference latency.

### Response (`200 OK`)
```json
{
  "total_scanned": 1542,
  "fraud_detected": 89,
  "fraud_rate_pct": 5.77,
  "avg_latency_ms": 7.42,
  "engine_status": "ONLINE"
}
```

---

## 3. Liveness & Model Health Check (`GET /health`)

Verifies memory integrity, PyTorch model loading status, and feature scaler availability.

### Response (`200 OK`)
```json
{
  "status": "ok",
  "gnn_model_loaded": true,
  "scaler_loaded": true,
  "uptime_seconds": 3840
}
```

---

## cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "U1024",
    "receiver_id": "U5001",
    "amount": 75000,
    "hour_of_day": 2,
    "is_weekend": 1
  }'
```
