FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model files from HF hub
RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hexgrad/Kokoro-82M-v1.0-ONNX', 'kokoro-v1.0.onnx', local_dir='/app/models'); hf_hub_download('hexgrad/Kokoro-82M-v1.0-ONNX', 'voices-v1.0.bin', local_dir='/app/models')"

# Copy application code
COPY . .

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
