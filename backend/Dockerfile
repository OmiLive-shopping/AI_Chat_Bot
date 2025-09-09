FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch separately due to custom index
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files into the container
COPY . .

# Set default port for Cloud Run
ENV PORT=8080

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
