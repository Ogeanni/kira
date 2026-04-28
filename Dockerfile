FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

EXPOSE 8000

# Startup sequence:
# 1. Pull synthetic data from S3
# 2. Run ingestion into Pinecone if index is empty
# 3. Start the FastAPI server
CMD ["sh", "-c", "python scripts/pull_data_from_s3.py && uvicorn output.api:app --host 0.0.0.0 --port 8000"]