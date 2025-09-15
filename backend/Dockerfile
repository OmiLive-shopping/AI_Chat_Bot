# Use a Python 3.10 slim base image for a smaller footprint.
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# --------------------------
# Install system dependencies
# --------------------------
# We keep build-essential and some core libs as they are often needed for some Python packages.
# This list is more minimal but should be sufficient.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Copy and install Python dependencies
# --------------------------
# COPY requirements.txt first to leverage Docker's build cache.
COPY requirements.txt .

# Install all Python dependencies from a single file.
# This is a cleaner approach than installing them manually.
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# Copy source code and data
# --------------------------
# Copy the entire source directory, including app.py, data/, and quizzes/.
COPY . .

# --------------------------
# Security hardening & permissions
# --------------------------
# Create a non-root user for security.
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories and assign ownership
RUN mkdir -p data quizzes \
    && chown -R appuser:appuser /app

# Switch to the non-root user.
USER appuser

# --------------------------
# Environment variables
# --------------------------
ENV PORT=8080
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

# --------------------------
# Health check
# --------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# --------------------------
# Run Gunicorn
# --------------------------
# The command remains the same, but the simplified setup makes it more robust.
CMD exec gunicorn --bind 0.0.0.0:$PORT \
    --workers ${GUNICORN_WORKERS:-2} \
    --threads ${GUNICORN_THREADS:-2} \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --log-level info \
    app:app