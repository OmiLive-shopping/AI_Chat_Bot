# --------------------------
# Base image
# --------------------------
FROM python:3.10-slim-bullseye

# --------------------------
# Set working directory
# --------------------------
WORKDIR /app

# --------------------------
# Install system dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Copy and install Python dependencies
# --------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# --------------------------
# Copy application code and assets
# --------------------------
COPY . .

# --------------------------
# Create non-root user and set permissions
# --------------------------
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
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
# Optional: build FAISS index inside container
# Uncomment if you want to build during image creation
# RUN python build_faiss.py

# --------------------------
# Expose port
# --------------------------
EXPOSE 8080

# --------------------------
# Run Gunicorn (production-ready)
# --------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "--capture-output", "--log-level", "info", "app:app"]
