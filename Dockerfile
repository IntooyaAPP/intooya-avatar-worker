FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy your project files
COPY handler.py /workspace/handler.py
COPY requirements.txt /workspace/requirements.txt

# If LatentSync is your repo, copy it in.
# If you keep it in GitHub, either COPY it from build context or clone it here.
COPY LatentSync /workspace/LatentSync

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

RUN mkdir -p /workspace/worker/output /workspace/worker/tmp

CMD ["python", "/workspace/handler.py"]
