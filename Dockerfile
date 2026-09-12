# Use lightweight official Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install curl for service healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and configuration
COPY vega/ /app/vega/
COPY config.yaml /app/config.yaml

# Expose API (8000) and Dashboard (8501) ports
EXPOSE 8000 8501

# Default command starts the API service
CMD ["uvicorn", "vega.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
