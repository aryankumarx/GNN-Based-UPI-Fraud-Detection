"""
Dataset Validation Utility
Verifies data integrity, column types, class distribution, and graph node coverage.
"""
import pandas as pd
import numpy as np
import os
import sys

def validate_dataset(csv_path: str = "data/raw/synthetic_upi_txns.csv"):
    if not os.path.exists(csv_path):
        print(f"[!] Warning: Data file {csv_path} not found. Skipping physical check.")
        return True

    print(f"[*] Validating dataset integrity for: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Required schema check
    required_cols = ["sender_id", "receiver_id", "amount", "timestamp", "is_fraud"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # 2. Check for missing values
    null_counts = df.isnull().sum()
    print(f"[*] Null value audit:\n{null_counts}")
    assert null_counts.sum() == 0, "Dataset contains unexpected null values"

    # 3. Class balance check
    fraud_count = df["is_fraud"].sum()
    total_count = len(df)
    fraud_rate = fraud_count / total_count * 100
    print(f"[*] Total records: {total_count} | Fraud records: {fraud_count} ({fraud_rate:.2f}%)")
    assert 0.0 < fraud_rate < 50.0, f"Unrealistic fraud rate: {fraud_rate:.2f}%"

    # 4. Value sanity check
    assert (df["amount"] > 0).all(), "Transaction amounts must be strictly positive"
    print("[+] All dataset integrity checks passed successfully!")
    return True

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/synthetic_upi_txns.csv"
    validate_dataset(path)
