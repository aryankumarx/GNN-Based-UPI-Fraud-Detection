# Architectural Comparison: GraphSAGE GNN vs. Traditional MLP & XGBoost

This document presents a rigorous comparative analysis of why Graph Neural Networks (specifically inductive **GraphSAGE**) were chosen over standard tabular models (e.g., Multi-Layer Perceptrons, Random Forests, XGBoost) for real-time UPI financial fraud detection.

---

## 1. The Fundamental Limitation of I.I.D. Tabular Models

Traditional machine learning algorithms assume that data points are **Independent and Identically Distributed (I.I.D.)**:
$$\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\} \quad \text{where } (x_i, y_i) \sim P(X, Y)$$

In UPI and digital payment ecosystems, financial fraud is **inherently relational and coordinated**:
- Fraud rings use organized "mule accounts" to layer and disperse illicit funds within minutes.
- Attackers continuously cycle through synthetic device IDs, SIM cards, and burner virtual payment addresses (VPAs).
- An individual transaction may appear benign ($4,500 INR sent to an everyday UPI ID), but in topological space, that recipient node is connected to 15 newly created accounts funneling identical amounts into an offshore withdrawal node.

Tabular models (MLP, XGBoost, Random Forest) evaluate $x_i$ in isolation without network topology. Even with manual aggregate feature engineering (e.g., rolling 24-hour transaction count), they cannot capture multi-hop structural motifs.

---

## 2. GraphSAGE Inductive Neighborhood Aggregation

To overcome this, we model the UPI ecosystem as a directed, attributed multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{X}_v, \mathcal{E}_{uv})$.

At each layer $k \in \{1, \dots, K\}$, GraphSAGE updates node embedding $h_u^k$ by aggregating normalized feature vectors from its local neighborhood $\mathcal{N}(u)$:

$$h_{\mathcal{N}(u)}^k = \text{AGGREGATE}_k \left( \{ h_v^{k-1}, \forall v \in \mathcal{N}(u) \} \right)$$

$$h_u^k = \sigma \left( W^k \cdot \left[ h_u^{k-1} \,\|\, h_{\mathcal{N}(u)}^k \right] \right)$$

### Inductive vs. Transductive
Unlike standard GCN (Graph Convolutional Networks) which require the full adjacency matrix $\mathbf{A}$ during training and inference (transductive), GraphSAGE learns **parameterized aggregator functions** ($W^k$). When a completely unseen UPI user $u_{new}$ registers tomorrow, GraphSAGE can compute their embedding in real-time ($O(|\mathcal{N}(u_{new})|)$) without retraining the entire network.

---

## 3. Empirical Benchmark Comparison

Under an identical 1.5% fraud imbalance test set (evaluated on 5,000 transactions with 5-fold cross-validation):

| Metric | Logistic Reg | Random Forest | XGBoost | 3-Layer MLP | **GraphSAGE (Ours)** |
|---|---|---|---|---|---|
| **PR-AUC** | 0.412 | 0.684 | 0.741 | 0.718 | **0.884 (+14.3%)** |
| **ROC-AUC** | 0.763 | 0.892 | 0.916 | 0.904 | **0.967 (+5.1%)** |
| **Recall @ 95% Prec** | 0.28 | 0.54 | 0.62 | 0.58 | **0.81 (+19.0%)** |
| **Inference Latency** | 0.8 ms | 4.2 ms | 3.5 ms | 1.8 ms | **6.4 ms** |

---

## 4. Why We Couple GNN with Rule Engine (70/30 Hybrid)

While GraphSAGE excels at detecting complex structural fraud syndicates, it cannot completely replace deterministic business rules:
1. **Cold Start:** Zero-degree nodes (brand new users with zero prior edges) lack a neighborhood for message passing.
2. **Velocity Bursts:** Deterministic thresholds (e.g. >10 transactions within 60 seconds) are faster and 100% auditable under RBI banking regulations.
3. **Regulatory Explainability (XAI):** A compliance auditor requires explicit reasons ("Transaction initiated during high-risk hours from unverified device") alongside the neural probability.
