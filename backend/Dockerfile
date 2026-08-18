FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ONNX model & voices at build time for instant zero-lag startup
RUN mkdir -p /app/models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hexgrad/Kokoro-82M-v1.0-ONNX', 'kokoro-v1.0.onnx', local_dir='/app/models'); hf_hub_download('hexgrad/Kokoro-82M-v1.0-ONNX', 'voices-v1.0.bin', local_dir='/app/models')"

# Copy application files
COPY . .

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
