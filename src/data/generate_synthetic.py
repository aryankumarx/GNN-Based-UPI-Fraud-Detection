"""
Synthetic UPI Transaction Dataset Generator.
Generates realistic UPI transactions with embedded fraud patterns:
- Mule account rings (multiple accounts sharing device/IP)
- Rapid transaction bursts (high velocity)
- Unusual high-value anomalies
- Device switching & geographical inconsistency
- Phishing / QR tampering patterns
"""

import os
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_upi_dataset(
    num_users: int = 5000,
    num_merchants: int = 500,
    num_transactions: int = 100000,
    fraud_rate: float = 0.02,
    output_path: str = "data/raw/upi_transactions_synthetic.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic UPI transaction records with realistic fraud networks."""
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Generating synthetic UPI dataset ({num_transactions} transactions)...")

    # 1. Banks & Handlers
    banks = ["SBI", "HDFC", "ICICI", "AXIS", "PNB", "BOB", "KOTAK", "PAYTM_BANK"]
    upi_handles = ["okhdfcbank", "oksbi", "okicici", "okaxis", "paytm", "apl", "ybl", "ibl"]
    device_types = ["Android", "iOS"]
    categories = ["P2P", "P2M", "Bill_Payment", "Recharge", "Online_Shopping", "Food_Dining", "Investment"]
    cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow"]

    # 2. Generate Entities (Users & Merchants)
    users = []
    for i in range(num_users):
        bank = random.choice(banks)
        handle = random.choice(upi_handles)
        vpa = f"user_{i:05d}@{handle}"
        primary_device = f"DEV_{random.randint(1000, 99999):05d}"
        primary_ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
        city = random.choice(cities)
        avg_amount = float(np.random.exponential(scale=800) + 50)
        
        users.append({
            "user_id": f"U_{i:05d}",
            "vpa": vpa,
            "bank": bank,
            "primary_device": primary_device,
            "primary_ip": primary_ip,
            "city": city,
            "avg_amount": avg_amount,
            "is_mule": False,
        })
    
    # Designate a cluster of mule accounts (fraud ring sharing devices/IPs)
    num_mules = int(num_users * 0.03)
    mule_indices = random.sample(range(num_users), num_mules)
    shared_fraud_devices = [f"DEV_MULE_{k:03d}" for k in range(10)]
    shared_fraud_ips = [f"10.0.99.{k}" for k in range(15)]
    
    for idx in mule_indices:
        users[idx]["is_mule"] = True
        users[idx]["primary_device"] = random.choice(shared_fraud_devices)
        users[idx]["primary_ip"] = random.choice(shared_fraud_ips)

    merchants = []
    for m in range(num_merchants):
        bank = random.choice(banks)
        handle = random.choice(["paytm", "bharatpe", "phonepe", "razorpay"])
        merchants.append({
            "merchant_id": f"M_{m:04d}",
            "vpa": f"merchant_{m:04d}@{handle}",
            "bank": bank,
            "category": random.choice(categories[1:]),
            "city": random.choice(cities),
        })

    # 3. Generate Transactions
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    transactions = []

    target_frauds = int(num_transactions * fraud_rate)
    normal_count = num_transactions - target_frauds

    print("Synthesizing normal transactions...")
    for _ in tqdm(range(normal_count), desc="Normal Txns"):
        sender = random.choice(users)
        is_p2m = random.random() < 0.45
        
        if is_p2m:
            receiver = random.choice(merchants)
            receiver_vpa = receiver["vpa"]
            receiver_bank = receiver["bank"]
            txn_type = "P2M"
            category = receiver["category"]
        else:
            rec_user = random.choice(users)
            while rec_user["user_id"] == sender["user_id"]:
                rec_user = random.choice(users)
            receiver_vpa = rec_user["vpa"]
            receiver_bank = rec_user["bank"]
            txn_type = "P2P"
            category = "P2P"

        amount = max(10.0, float(np.random.normal(loc=sender["avg_amount"], scale=sender["avg_amount"] * 0.4)))
        
        # Realistic timestamp (Gaussian centered around daytime)
        hour = int(np.clip(np.random.normal(loc=14, scale=4), 0, 23))
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        day_offset = random.randint(0, 60)
        timestamp = start_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        device = sender["primary_device"]
        ip = sender["primary_ip"]
        location = sender["city"]

        transactions.append({
            "transaction_id": str(uuid.uuid4())[:18],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_id": sender["user_id"],
            "sender_vpa": sender["vpa"],
            "sender_bank": sender["bank"],
            "receiver_vpa": receiver_vpa,
            "receiver_bank": receiver_bank,
            "amount": round(amount, 2),
            "transaction_type": txn_type,
            "category": category,
            "device_id": device,
            "device_type": random.choice(device_types),
            "ip_address": ip,
            "location": location,
            "is_fraud": 0,
            "fraud_type": "None",
        })

    print("Synthesizing fraud patterns...")
    fraud_types = ["Mule_Ring_Collusion", "Velocity_Burst", "High_Value_Anomaly", "Device_Hijack_Phishing"]
    
    for _ in tqdm(range(target_frauds), desc="Fraud Txns"):
        ftype = random.choice(fraud_types)
        day_offset = random.randint(0, 60)
        
        if ftype == "Mule_Ring_Collusion":
            # Transactions between mule accounts sharing device/IP
            sender = users[random.choice(mule_indices)]
            receiver = users[random.choice(mule_indices)]
            amount = round(float(random.uniform(10000, 95000)), 2)
            timestamp = start_time + timedelta(days=day_offset, hours=random.choice([1, 2, 3, 4, 23]), minutes=random.randint(0, 59))
            device = random.choice(shared_fraud_devices)
            ip = random.choice(shared_fraud_ips)
            txn_type = "P2P"
            category = "P2P"
            rec_vpa = receiver["vpa"]
            rec_bank = receiver["bank"]
            loc = sender["city"]

        elif ftype == "Velocity_Burst":
            # Rapid high frequency transactions in seconds/minutes
            sender = random.choice(users)
            receiver = users[random.choice(mule_indices)]
            amount = round(float(random.uniform(5000, 20000)), 2)
            base_time = start_time + timedelta(days=day_offset, hours=random.randint(0, 23), minutes=random.randint(0, 50))
            timestamp = base_time + timedelta(seconds=random.randint(1, 45))
            device = sender["primary_device"]
            ip = sender["primary_ip"]
            txn_type = "P2P"
            category = "P2P"
            rec_vpa = receiver["vpa"]
            rec_bank = receiver["bank"]
            loc = sender["city"]

        elif ftype == "High_Value_Anomaly":
            # Unusual 10x-50x spike in amount at midnight
            sender = random.choice(users)
            receiver = random.choice(merchants)
            amount = round(float(sender["avg_amount"] * random.uniform(15, 60)), 2)
            amount = min(amount, 100000.0)  # UPI limit ₹1 Lakh
            timestamp = start_time + timedelta(days=day_offset, hours=random.choice([0, 1, 2, 3]), minutes=random.randint(0, 59))
            device = f"DEV_UNKNOWN_{random.randint(100, 999)}"
            ip = f"45.33.{random.randint(1, 254)}.{random.randint(1, 254)}"
            txn_type = "P2M"
            category = "Investment"
            rec_vpa = receiver["vpa"]
            rec_bank = receiver["bank"]
            loc = random.choice(cities)

        else:  # Device_Hijack_Phishing
            sender = random.choice(users)
            receiver = users[random.choice(mule_indices)]
            amount = round(float(random.uniform(25000, 99000)), 2)
            timestamp = start_time + timedelta(days=day_offset, hours=random.randint(9, 22), minutes=random.randint(0, 59))
            device = random.choice(shared_fraud_devices)
            ip = random.choice(shared_fraud_ips)
            txn_type = "P2P"
            category = "P2P"
            rec_vpa = receiver["vpa"]
            rec_bank = receiver["bank"]
            loc = random.choice([c for c in cities if c != sender["city"]])

        transactions.append({
            "transaction_id": str(uuid.uuid4())[:18],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_id": sender["user_id"],
            "sender_vpa": sender["vpa"],
            "sender_bank": sender["bank"],
            "receiver_vpa": receiver_vpa,
            "receiver_bank": receiver_bank,
            "amount": amount,
            "transaction_type": txn_type,
            "category": category,
            "device_id": device,
            "device_type": random.choice(device_types),
            "ip_address": ip,
            "location": loc,
            "is_fraud": 1,
            "fraud_type": ftype,
        })

    df = pd.DataFrame(transactions)
    # Sort chronologically
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp_dt").reset_index(drop=True)
    df = df.drop(columns=["timestamp_dt"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully! Total rows: {len(df)}, Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    generate_upi_dataset()
