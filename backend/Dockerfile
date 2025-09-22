# Use a Python 3.10 slim base image for a smaller footprint
# Explicitly use the linux/amd64 platform to ensure compatibility with Cloud Run
FROM --platform=linux/amd64 python:3.10-slim-bullseye

# Set working directory in the container
WORKDIR /app

# --------------------------
# Install system dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Copy and install Python dependencies
# --------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# Copy source code and data
# --------------------------
COPY . .

# --------------------------
# Security hardening & permissions
# --------------------------
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p data quizzes \
    && chown -R appuser:appuser /app

USER appuser

# --------------------------
# Environment variables
# --------------------------
# Google Cloud Run will inject credentials automatically
ENV PORT=8080
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV GOOGLE_APPLICATION_CREDENTIALS=""

# --------------------------
# Health check
# --------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# --------------------------
# Run Gunicorn
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