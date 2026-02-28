# Multi-stage Dockerfile for ragdag
# Stage 1: Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build essentials for any compiled deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY sdk/ sdk/

# Install Python package with server + mcp extras
RUN pip install --no-cache-dir --prefix=/install ".[server,mcp]"

# Stage 2: Runtime
FROM python:3.11-slim

# Install bash + coreutils for ragdag CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    coreutils \
    grep \
    gawk \
    ripgrep \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY ragdag ./ragdag
COPY lib/ ./lib/
COPY engines/ ./engines/
COPY server/ ./server/
COPY sdk/ ./sdk/
COPY pyproject.toml .

# Make CLI executable
RUN chmod +x ./ragdag

# Set up paths
ENV RAGDAG_DIR=/app
ENV PYTHONPATH=/app:/app/sdk
ENV PATH="/app:${PATH}"

# Default store location (mount a volume here)
ENV RAGDAG_STORE=/data
RUN mkdir -p /data

EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/health')" || exit 1

# Default: start HTTP API server
CMD ["python3", "-c", "import sys; sys.path.insert(0, '/app'); sys.path.insert(0, '/app/sdk'); from server.api import run; run(host='0.0.0.0', port=8420)"]
