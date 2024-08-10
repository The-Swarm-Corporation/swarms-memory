# Start with the official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_HOST="0.0.0.0"
ENV API_PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies and update package list
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application with Uvicorn and Gunicorn
CMD ["gunicorn", "main:api", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
