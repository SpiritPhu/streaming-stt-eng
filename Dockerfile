# Production Dockerfile for Realtime Streaming Speech-to-Text
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (curl for healthchecks, libgomp1 for onnxruntime/sherpa-onnx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-privileged user and default application directories
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/data/recordings /app/models/zipformer /app/static && \
    chown -R appuser:appuser /app

# Install Python dependencies first for efficient layer caching
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and static web assets
COPY --chown=appuser:appuser app.py db.py ./
COPY --chown=appuser:appuser static/ ./static/

# Switch to non-root user
USER appuser

# Expose HTTP & WebSocket port
EXPOSE 8000

# Periodic health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Production entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
