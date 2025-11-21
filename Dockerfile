# Use PyTorch CPU image to avoid building heavy ML packages from source.
# This image includes a compatible Python and prebuilt PyTorch wheels.
# Using bookworm instead of bullseye for SQLite 3.35.0+ support required by ChromaDB
FROM python:3.11-slim-bookworm

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install small set of build deps required by some packages
# Including Playwright browser dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        procps \
       build-essential \
       git \
       curl \
       gcc \
        pkg-config \
        libgomp1 \
        libopenblas-dev \
        libsndfile1 \
        ffmpeg \
       libffi-dev \
       libssl-dev \
        # Playwright browser dependencies
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libatspi2.0-0 \
        libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
# Install CPU PyTorch wheels first from the official PyTorch wheel index to avoid source builds
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio || true \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt \
    && pip install playwright \
    && playwright install chromium \
    && playwright install-deps chromium

# Copy project files
COPY . /app

# Ensure chroma path exists
RUN mkdir -p /app/temp_chroma || true

# Expose server port
EXPOSE 8000

# Default command runs the FastAPI app via uvicorn
CMD ["uvicorn", "rest_api:app", "--host", "0.0.0.0", "--port", "8000"]
