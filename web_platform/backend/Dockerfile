FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MedRAX dependencies (will be mounted as volume)
# The actual MedRAX code will be mounted as a volume

COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]



