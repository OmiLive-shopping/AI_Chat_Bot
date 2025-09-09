FROM python:3.10-slim

WORKDIR /app

# --------------------------
# System dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    libopenblas-dev \
    libomp-dev \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Python dependencies
# --------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    gunicorn==21.2.0 \
    prometheus-flask-exporter==0.22.4

# --------------------------
# Copy source code
# --------------------------
COPY . .

# --------------------------
# Security hardening
# --------------------------
# Create a non-root user BEFORE assigning ownership
RUN groupadd -r appuser && useradd -r -g appuser appuser

# --------------------------
# Create necessary directories and assign ownership
RUN mkdir -p data \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser/.cache \
    && chown -R appuser:appuser /app

# --------------------------
# Environment variables
# --------------------------
ENV PORT=8080
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

USER appuser

# --------------------------
# Health check
# --------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# --------------------------
# Run Gunicorn with optimized settings
# --------------------------
CMD exec gunicorn --bind 0.0.0.0:$PORT \
    --workers ${GUNICORN_WORKERS:-2} \
    --threads ${GUNICORN_THREADS:-2} \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --log-level info \
    app:app
