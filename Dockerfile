FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build so they're cached in the image
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
 SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
 CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy source code
COPY . .

# Create required directories
RUN mkdir -p storage/uploads storage/index

# Health check (matches the dynamic port or default 8000)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
  CMD python -c "import urllib.request; import os; port=os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

EXPOSE 8000

# Use shell form to allow environment variable expansion for Render/Cloud platforms
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
