# Contributing to GNN-Based UPI Fraud Detection

Thank you for your interest in contributing! We welcome community contributions, bug reports, and research enhancements.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/GNN-Based-UPI-Fraud-Detection.git
   cd GNN-Based-UPI-Fraud-Detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt          # Lightweight runtime & deployment
   pip install -r requirements-train.txt    # Full local ML & GNN research
   ```

4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

## Contribution Workflow

1. Create a feature branch (`git checkout -b feat/your-feature-name`).
2. Adhere to [PEP 8](https://peps.python.org/pep-0008/) style standards.
3. Add unit tests under `tests/` for all newly introduced pipeline components or loss functions.
4. Verify all tests pass with `pytest`.
5. Submit a descriptive Pull Request with benchmark comparisons if model weights are modified.

## Code Style & Commit Guidelines
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `perf:`, `chore:`.
- Ensure latency is preserved (< 10 ms per transaction).
