# Pure Sound - Professional Audio Processing Suite
# Multi-stage build for optimized image size and security

# Stage 1: Builder - Install dependencies and build the application
FROM python:3.13-slim-bullseye as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    libopus0 \
    libmp3lame0 \
    libfdk-aac2 \
    libsamplerate0-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir numpy scipy boto3 PyQt6

# Copy application code
COPY . .

# Install application as user for security
RUN chown -R appuser:appuser /app

# Stage 2: Final image - Minimal runtime environment
FROM python:3.13-slim-bullseye as final

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopus0 \
    libmp3lame0 \
    libfdk-aac2 \
    libsamplerate0 \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --chown=appuser:appuser . .

# Create directories for outputs and models
RUN mkdir -p /app/outputs /app/models /app/config && \
    chown -R appuser:appuser /app/outputs /app/models /app/config

# Switch to non-root user for security
USER appuser

# Expose port (if web interface is added later)
EXPOSE 8080

# Set default command
CMD ["python", "compress_audio.py", "--help"]