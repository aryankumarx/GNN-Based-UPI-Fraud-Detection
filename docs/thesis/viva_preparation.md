# Project Viva & Academic Defense Master Guide

This guide compiles high-frequency questions, mathematical defenses, and architectural rationales for academic viva examinations and project evaluations.

---

## 1. Core Problem Statement

### Q: Why do traditional fraud detection systems fail on UPI?
**Answer:**
Traditional fraud detection engines rely heavily on **I.I.D. (Independent and Identically Distributed)** tabular models (e.g., Logistic Regression, Random Forest, XGBoost) and static rule sets. 

In UPI, modern fraud operates through **coordinated mule rings** and **network layering**:
1. Stolen funds are rapidly split into micro-transactions ($5,000 - $10,000 INR) across dozens of proxy accounts within seconds to avoid high-value thresholds.
2. An individual transaction in isolation appears completely legitimate.
3. Only by inspecting the **k-hop topological graph structure** can an engine observe that multiple sender nodes funnel into a single unverified cash-out node.

---

## 2. Graph Neural Network & GraphSAGE

### Q: Why GraphSAGE instead of standard GCN (Graph Convolutional Networks)?
**Answer:**
Standard GCNs are **transductive**: they require the complete graph adjacency matrix $\mathbf{A}$ during both training and inference. In a real-time payment gateway where millions of new users and transactions arrive dynamically, recalculating the entire graph adjacency matrix is computationally impossible ($O(N^2)$).

**GraphSAGE is inductive**:
1. Instead of learning fixed node embeddings, it learns **parameterized aggregator functions** ($W^k$).
2. During inference for an incoming transaction, GraphSAGE only samples the $k$-hop neighborhood of the sender and receiver ($\mathcal{N}(u)$).
3. It can compute embeddings for completely unseen nodes on the fly in sub-10 milliseconds without full graph retraining.

---

## 3. Class Imbalance & Focal Loss

### Q: In UPI transactions, fraud is typically < 1%. How do you handle this severe class imbalance?
**Answer:**
Standard Binary Cross-Entropy (BCE) loss suffers because the overwhelming majority of benign transactions dominate the gradient:

$$\mathcal{L}_{BCE} = - [y \log(p) + (1-y) \log(1-p)]$$

We utilize **Sigmoid Focal Loss** with modulating factor $\gamma = 2.0$ and weighting factor $\alpha = 0.25$:

$$\mathcal{L}_{FL} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

- For well-classified benign examples ($p_t \approx 0.99$), the modulating factor $(1 - p_t)^2 \approx 0.0001$, scaling down their gradient contribution by $10,000\times$.
- For hard, misclassified fraud examples ($p_t \approx 0.2$), the modulating factor $(1 - p_t)^2 \approx 0.64$, focusing model parameter updates almost entirely on anomalous transactions.

---

## 4. The 70/30 Hybrid Decision Engine

### Q: Why not use 100% GNN? Why combine it with a Rule Engine?
**Answer:**
1. **Cold-Start Nodes:** When a brand-new user conducts their very first transaction, their graph degree is 0. Neighborhood aggregation yields an uninformative prior. The rule engine covers this blind spot.
2. **Instant Velocity Exploits:** Rule engines can compute millisecond sliding-window velocities (e.g., >10 transactions/minute) directly from Redis counters without waiting for graph edge formation.
3. **Regulatory Explainability (RBI Compliance):** Financial regulators mandate that rejected payments have clear explanations. The rule engine provides deterministic audit tags (e.g., `"Nocturnal burst from unverified device"`).

$$\text{Hybrid Score} = 0.70 \times P_{\text{GNN}} + 0.30 \times S_{\text{Rule}}$$

---

## 5. Performance Benchmarks

- **Mean Inference Latency:** 6.4 ms (target: <50 ms SLA).
- **PR-AUC:** 0.884 (+14.3% higher than XGBoost).
- **ROC-AUC:** 0.967.
- **Recall @ 95% Precision:** 81%.
