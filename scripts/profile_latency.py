#!/usr/bin/env python3
"""
Performance & Latency Benchmark Profiler
Measures single-batch and multi-threaded throughput and percentiles (P50, P95, P99).
"""
import time
import numpy as np
from src.pipeline.hybrid_pipeline import HybridPipeline

def profile_engine(iterations: int = 200):
    print(f"[*] Initializing Hybrid Fraud Detection Engine...")
    engine = HybridPipeline()
    latencies = []

    test_txns = [
        {"sender_id": "U1001", "receiver_id": "U9999", "amount": 1500.0, "hour_of_day": 12, "is_weekend": 0, "device_id": "DEV_A", "ip_address": "1.1.1.1", "location": "DELHI"},
        {"sender_id": "U2002", "receiver_id": "U8888", "amount": 95000.0, "hour_of_day": 3, "is_weekend": 1, "device_id": "DEV_B", "ip_address": "1.1.1.2", "location": "MUMBAI"},
        {"sender_id": "U3003", "receiver_id": "U7777", "amount": 420.0, "hour_of_day": 18, "is_weekend": 0, "device_id": "DEV_C", "ip_address": "1.1.1.3", "location": "BANGALORE"},
    ]

    print(f"[*] Warming up engine with 20 transactions...")
    for _ in range(20):
        engine.predict(test_txns[0])

    print(f"[*] Running latency benchmark over {iterations} iterations...")
    for i in range(iterations):
        sample = test_txns[i % len(test_txns)]
        t0 = time.perf_counter()
        _ = engine.predict(sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    print("\n" + "="*50)
    print("  LATENCY & THROUGHPUT BENCHMARK RESULTS")
    print("="*50)
    print(f"  Total Samples  : {iterations}")
    print(f"  Mean Latency   : {np.mean(arr):.2f} ms")
    print(f"  Median (P50)   : {np.percentile(arr, 50):.2f} ms")
    print(f"  P95 Latency    : {np.percentile(arr, 95):.2f} ms")
    print(f"  P99 Latency    : {np.percentile(arr, 99):.2f} ms")
    print(f"  Throughput     : {1000.0 / np.mean(arr):.1f} txns/sec (Single Core)")
    print("="*50 + "\n")

if __name__ == "__main__":
    profile_engine()
