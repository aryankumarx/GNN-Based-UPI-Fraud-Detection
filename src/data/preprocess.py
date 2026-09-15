"""
Data preprocessing, feature engineering, and temporal splitting for UPI Fraud Detection.
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class UPIPreprocessor:
    """Preprocesses UPI transaction data and prepares node/edge features."""

    def __init__(self, data_path: str = "data/raw/upi_transactions_synthetic.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def load_data(self) -> pd.DataFrame:
        """Load and parse transaction data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def compute_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute temporal and behavioural features per transaction."""
        df = df.copy()

        # 1. Temporal components
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(int)

        # 2. Log-transformed amount
        df["log_amount"] = np.log1p(df["amount"])

        # 3. User transaction velocity (rolling transaction counts in last 1 hour)
        df = df.sort_values("timestamp")
        
        # User transaction count and rolling mean amount
        user_stats = df.groupby("sender_id").agg(
            user_mean_amount=("amount", "mean"),
            user_std_amount=("amount", "std"),
            user_txn_count=("transaction_id", "count")
        ).reset_index()
        user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0)

        df = df.merge(user_stats, on="sender_id", how="left")
        
        # Amount anomaly score: z-score against user's historical transactions
        df["amount_zscore"] = np.where(
            df["user_std_amount"] > 0,
            (df["amount"] - df["user_mean_amount"]) / (df["user_std_amount"] + 1e-5),
            0.0
        )
        df["amount_zscore"] = np.clip(df["amount_zscore"], -5.0, 15.0)

        return df

    def prepare_tabular_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Encode categorical and scale numerical features for baseline models."""
        df_feat = self.compute_engineered_features(df)

        categorical_cols = [
            "sender_bank",
            "receiver_bank",
            "transaction_type",
            "category",
            "device_type",
            "location",
        ]
        
        for col in categorical_cols:
            if col in df_feat.columns:
                le = LabelEncoder()
                df_feat[col + "_enc"] = le.fit_transform(df_feat[col].astype(str))
                self.label_encoders[col] = le

        numerical_cols = [
            "amount",
            "log_amount",
            "hour",
            "day_of_week",
            "is_night",
            "user_mean_amount",
            "amount_zscore",
            "user_txn_count",
        ] + [c + "_enc" for c in categorical_cols if c in df_feat.columns]

        X = df_feat[numerical_cols].values
        y = df_feat["is_fraud"].values.astype(int)

        return X, y, df_feat

    def temporal_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """Perform strict chronological train / validation / test split."""
        n = len(X)
        n_train = int(n * train_ratio)
        n_val = int(n * (train_ratio + val_ratio))

        return {
            "X_train": X[:n_train],
            "y_train": y[:n_train],
            "X_val": X[n_train:n_val],
            "y_val": y[n_train:n_val],
            "X_test": X[n_val:],
            "y_test": y[n_val:],
        }


if __name__ == "__main__":
    preprocessor = UPIPreprocessor()
    df = preprocessor.load_data()
    print(f"Loaded {len(df)} transactions.")
    X, y, df_features = preprocessor.prepare_tabular_features(df)
    splits = preprocessor.temporal_split(X, y)
    print(f"Train size: {len(splits['X_train'])} (Fraud: {splits['y_train'].sum()})")
    print(f"Val size:   {len(splits['X_val'])} (Fraud: {splits['y_val'].sum()})")
    print(f"Test size:  {len(splits['X_test'])} (Fraud: {splits['y_test'].sum()})")
