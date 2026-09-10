# GNN-Based UPI Fraud Detection

> **Project 709** — Graph Neural Networks for Real-Time UPI Transaction Fraud Detection

## Overview

A Graph Neural Network (GNN) system that models UPI transactions as a dynamic heterogeneous graph and detects fraudulent transactions in real-time. The system leverages relational structure between users, merchants, devices, and accounts to catch fraud patterns that traditional ML models miss.

## Tech Stack

| Layer | Technology |
|---|---|
| **GNN Model** | Python, PyTorch, PyTorch Geometric |
| **Data Processing** | Pandas, NetworkX, NumPy |
| **Explainability** | GNNExplainer, SHAP |
| **Backend API** | FastAPI |
| **Dashboard** | Streamlit, PyVis, Plotly |
| **Database** | SQLite |

## Project Structure

```
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed graph data (.pt files)
│   └── README.md
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Feature engineering & cleaning
│   │   └── graph_builder.py    # Graph construction
│   ├── models/
│   │   ├── graphsage.py        # GraphSAGE model
│   │   ├── gat.py              # GAT model
│   │   └── baselines.py        # XGBoost, MLP baselines
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   ├── evaluate.py         # Evaluation metrics
│   │   └── losses.py           # Focal loss, etc.
│   ├── inference/
│   │   ├── pipeline.py         # Real-time inference pipeline
│   │   └── rules.py            # Rule-based fraud checks
│   ├── explainability/
│   │   └── explain.py          # GNNExplainer / SHAP integration
│   └── utils/
│       ├── config.py           # Hyperparameters & config
│       └── visualization.py    # Graph plotting utilities
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_graph_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_results.ipynb
├── api/
│   └── main.py                 # FastAPI application
├── configs/
│   └── default.yaml            # Training config
├── tests/
├── models/                     # Saved model checkpoints
├── docs/
│   ├── report/
│   └── presentation/
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Datasets

- **Primary**: [Kaggle UPI Transactions 2024](https://www.kaggle.com/datasets/skullagos5246/upi-transactions-2024-dataset)
- **Benchmark**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

## Usage

```bash
# Train model
python src/training/train.py

# Start API
uvicorn api.main:app --reload

# Launch dashboard
streamlit run dashboard/app.py
```

## Key Results

| Model | Precision | Recall | F1 | AUPRC |
|---|---|---|---|---|
| XGBoost (baseline) | — | — | — | — |
| MLP (baseline) | — | — | — | — |
| GraphSAGE | — | — | — | — |
| GAT | — | — | — | — |
| **GNN + Rules (Hybrid)** | — | — | — | — |

*Results will be updated as experiments complete.*

## Team

- Final Year Project — 2026-27

## License

MIT
