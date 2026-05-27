# Dockerfile — Containerised deployment for the Customer Support LLM API
# 
# Build:  docker build -t customer-support-llm .
# Run:    docker run --gpus all -p 8000:8000 -v ./model:/app/model customer-support-llm
#
# Requires NVIDIA Container Toolkit for GPU access:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY config.py .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
