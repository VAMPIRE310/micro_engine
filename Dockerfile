# ============================================================
# Dockerfile — Python Micro-Execution Engine
# Railway / Docker deployment
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# System deps required for pycryptodome, cryptography, and
# building any native extensions (gcc, libssl-dev, pkg-config).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — leverage Docker layer cache
COPY requirements.txt .

# Install Python deps (CPU-only torch saves ~1 GB vs default CUDA wheel)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Python runtime tunables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

# Default: run the main engine
# Override CMD in docker-compose or Railway service settings if needed.
CMD ["python", "main_micro_engine.py"]
