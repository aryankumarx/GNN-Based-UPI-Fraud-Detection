#!/usr/bin/env python3
"""
CLI Transaction Tester
Quickly test arbitrary UPI transactions against the detection pipeline or live server.
Usage:
    python scripts/test_txn_cli.py --amount 85000 --sender U1001 --receiver U9999 --hour 3
"""
import argparse
import json
import urllib.request

def run_test(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/predict",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            print("\n" + "="*50)
            print("  UPI FRAUD DETECTION RESULT")
            print("="*50)
            print(f"  Decision       : {res.get('action')}")
            print(f"  Is Fraud       : {res.get('is_fraud')}")
            print(f"  Hybrid Score   : {res.get('hybrid_score'):.4f} (Threshold: {res.get('decision_threshold')})")
            print(f"  GNN Probability: {res.get('gnn_prob'):.4f}")
            print(f"  Rule Score     : {res.get('rule_score'):.4f}")
            print(f"  Rule Reasons   : {', '.join(res.get('rule_reasons', [])) or 'None'}")
            print(f"  Latency        : {res.get('latency_ms', 0):.2f} ms")
            print("="*50 + "\n")
    except Exception as e:
        print(f"[!] Error calling API at {url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test UPI Transaction against Fraud Engine")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--sender", default="U1001", help="Sender VPA / ID")
    parser.add_argument("--receiver", default="U9999", help="Receiver VPA / ID")
    parser.add_argument("--amount", type=float, default=25000.0, help="Transaction amount in INR")
    parser.add_argument("--hour", type=int, default=14, help="Hour of transaction (0-23)")
    parser.add_argument("--weekend", type=int, default=0, help="Is weekend (0 or 1)")
    parser.add_argument("--device", default="DEV_TEST_01", help="Device identifier")
    parser.add_argument("--location", default="DELHI", help="Geographic location")

    args = parser.parse_args()

    payload = {
        "sender_id": args.sender,
        "receiver_id": args.receiver,
        "amount": args.amount,
        "hour_of_day": args.hour,
        "is_weekend": args.weekend,
        "device_id": args.device,
        "ip_address": "192.168.1.100",
        "location": args.location
    }

    run_test(args.url, payload)
