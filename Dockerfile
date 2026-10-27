# Dockerfile for Blockchain-Enabled Secure FL for IIoT
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python packages (CPU PyTorch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run preliminary experiment to ensure pre-warmed models and results
RUN python src/run_experiment.py --rounds 3 --epochs 1 || true

# Expose Streamlit default port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
