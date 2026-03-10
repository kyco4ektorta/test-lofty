FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.10 /usr/bin/python3 && ln -sf pip3 /usr/bin/pip

WORKDIR /app

# Python deps
COPY requirements_worker.txt .
RUN pip install --no-cache-dir -r requirements_worker.txt

# Pre-download model weights (baked into image for faster cold start)
ARG MODEL_SIZE=large
RUN python3 -c "from audiocraft.models import MusicGen; MusicGen.get_pretrained('facebook/musicgen-${MODEL_SIZE}')" \
    || echo "Model pre-download skipped"

COPY handler.py .

# Expose for health checks
EXPOSE 8000

CMD ["python3", "-u", "handler.py"]
