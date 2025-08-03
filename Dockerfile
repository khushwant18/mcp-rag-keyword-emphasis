FROM python:3.11-slim

# Install system dependencies including Chromium for crawl4ai
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    chromium \
    chromium-driver \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model during build
RUN python -m playwright install 
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1',trust_remote_code=True)"

# Copy application code
COPY mcp_server.py .

# Create necessary directories
RUN mkdir -p /data/chroma_db /logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROMA_DB_PATH=/data/chroma_db

# Run the server
CMD ["python", "mcp_server.py", "--http", "--port", "8000"]