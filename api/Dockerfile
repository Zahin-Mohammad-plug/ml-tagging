FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: tag_prompts.json will be available via volume mount at /app/workers/tag_prompts.json
# or we can copy it if building from parent context
# For now, the tag_sync service will look for it in multiple locations

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r tagger && useradd -r -g tagger tagger
RUN chown -R tagger:tagger /app
USER tagger

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]