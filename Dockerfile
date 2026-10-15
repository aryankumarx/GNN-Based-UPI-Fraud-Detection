# Multi-stage lightweight Dockerfile for GNN UPI Fraud Detection Service
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first for minimal container size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and core dependencies
RUN pip install --no-cache-dir \
    torch-geometric \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    pyyaml

# Copy application files
COPY . /app

# Expose port
EXPOSE 8000

# Start command
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
