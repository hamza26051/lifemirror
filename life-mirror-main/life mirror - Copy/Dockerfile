# =============================================================================
# Life Mirror - AI-Powered Self Analysis App
# Production-Ready Dockerfile
# =============================================================================
# Multi-stage build for optimal production deployment
# Supports both backend API and frontend web components

# =============================================================================
# STAGE 1: Base Image with System Dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    # Image processing dependencies
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Network and security
    curl \
    wget \
    ca-certificates \
    gnupg \
    # File system utilities
    unzip \
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# =============================================================================
# STAGE 2: Python Dependencies
# =============================================================================
FROM base as python-deps

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# =============================================================================
# STAGE 3: Node.js Dependencies
# =============================================================================
FROM node:18-alpine as node-deps

# Set working directory
WORKDIR /app

# Copy package files
COPY upload-image-app/package*.json ./upload-image-app/

# Install Node.js dependencies
RUN cd upload-image-app && \
    npm ci --only=production --silent

# =============================================================================
# STAGE 4: Application Build
# =============================================================================
FROM base as build

# Set working directory
WORKDIR /app

# Copy Python dependencies from previous stage
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy Node.js dependencies from previous stage
COPY --from=node-deps /app/upload-image-app/node_modules ./upload-image-app/node_modules

# Copy application source code
COPY . .

# Download YOLO models if not present
RUN if [ ! -f yolov8n.pt ]; then \
        wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt; \
    fi && \
    if [ ! -f yolov8n-pose.pt ]; then \
        wget -O yolov8n-pose.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt; \
    fi

# Set proper permissions
RUN chown -R appuser:appuser /app

# =============================================================================
# STAGE 5: Production Runtime
# =============================================================================
FROM base as production

# Set working directory
WORKDIR /app

# Copy Python runtime and dependencies
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy application code and models
COPY --from=build /app .

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/temp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Set environment variables for production
ENV FLASK_ENV=production \
    FLASK_APP=lifemirror_api.py \
    PYTHONPATH=/app

# Default command
CMD ["python", "lifemirror_api.py"]

# =============================================================================
# STAGE 6: Development Environment (Optional)
# =============================================================================
FROM base as development

# Set working directory
WORKDIR /app

# Copy all dependencies and source
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app .

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    black==23.12.1 \
    flake8==7.0.0 \
    isort==5.13.2 \
    mypy==1.8.0

# Set development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=1 \
    FLASK_APP=lifemirror_api.py

# Expose ports for development
EXPOSE 5000 8000

# Development command with auto-reload
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000", "--reload"]

# =============================================================================
# STAGE 7: Multi-Service Setup (Optional)
# =============================================================================
FROM base as multi-service

# Set working directory
WORKDIR /app

# Copy all components
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app .

# Install additional services
RUN pip install --no-cache-dir \
    gunicorn==21.2.0 \
    uvicorn==0.24.0 \
    supervisor==4.2.5

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log directories
RUN mkdir -p /var/log/supervisor /app/logs

# Expose multiple ports
EXPOSE 5000 8000 9000

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 