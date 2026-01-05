# Dockerfile for RegCompliance Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Hugging Face model during build (network available during build)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/import /app/form /app/data

# Expose ports for all services
# PolicyAPI: 8000, ProcessFlow: 6868, JudgeFlow: 6969
EXPOSE 8000 6868 6969

# Default command (can be overridden in docker-compose)
CMD ["python", "PolicyAPI.py"]
