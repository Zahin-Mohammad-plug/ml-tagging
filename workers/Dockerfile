# Multi-stage Dockerfile for workers
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1 \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# openai-whisper requires pkg_resources at build time, which was removed in setuptools>=78.
# Pin setuptools<78 so pkg_resources remains available.
RUN pip install --no-cache-dir --upgrade pip "setuptools<78" wheel && \
    pip install --no-cache-dir --no-build-isolation openai-whisper==20240930 && \
    pip install --no-cache-dir -r requirements.txt

# Copy worker code (this layer will be invalidated when code changes)
COPY . .

# Create non-root user
RUN groupadd -r worker && useradd -r -g worker worker
RUN chown -R worker:worker /app

# Create cache directory with proper permissions
RUN mkdir -p /tmp/cache && chown -R worker:worker /tmp/cache
RUN mkdir -p /app/.cache/frames && chown -R worker:worker /app/.cache

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Don't switch to worker user here - entrypoint will handle it
# USER worker

# Base command (will be overridden by specific worker targets)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", "-l", "info"]

# Sampler worker (FFmpeg frame extraction)
FROM base as sampler
ENV WORKER_TYPE=sampler
CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", "-Q", "sampling", "-n", "sampler@%h", "-l", "info"]

# Embeddings worker (GPU-intensive)
FROM base as embeddings
ENV WORKER_TYPE=embeddings
CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", "-Q", "embeddings", "-n", "embeddings@%h", "-l", "info", "--concurrency=1"]

# ASR/OCR worker
FROM base as asr-ocr
ENV WORKER_TYPE=asr_ocr
CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", "-Q", "asr_ocr", "-n", "asr_ocr@%h", "-l", "info"]

# Fusion worker (combines signals and generates suggestions)
FROM base as fusion
ENV WORKER_TYPE=fusion
CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", "-Q", "fusion", "-n", "fusion@%h", "-l", "info"]