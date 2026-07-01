# Outfit Completion Pipeline — Dockerfile
FROM python:3.11-slim

# System deps for torch/pillow/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer — speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Data and output directories (mounted as volumes, see docker-compose.yml)
RUN mkdir -p /app/data /app/outputs

ENV PYTHONUNBUFFERED=1
ENV CSV_PATH=/app/data/modanisa_products.csv
ENV QDRANT_PATH=/app/data/qdrant_storage
ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_TEXT_MODEL=mistral

WORKDIR /app/app

ENTRYPOINT ["python", "outfit_pipeline.py"]
CMD ["--help"]
