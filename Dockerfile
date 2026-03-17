FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace/MuseTalk"

# System deps (ADD wget here)
RUN apt-get update && apt-get install -y ffmpeg git wget

WORKDIR /workspace

# Prebuilt torch (fast)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OpenMMLab (required)
RUN pip install --no-cache-dir openmim
RUN mim install "mmcv>=2.0.0"
RUN mim install "mmpose>=1.0.0"

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git

WORKDIR /workspace/MuseTalk

# Install MuseTalk deps
RUN pip install --no-cache-dir -r requirements.txt

# 🔥 DOWNLOAD REQUIRED MODELS
RUN mkdir -p /workspace/MuseTalk/models/dwpose
RUN mkdir -p /workspace/MuseTalk/models/musetalkV15
RUN mkdir -p /workspace/MuseTalk/models/sd-vae

# dwpose model
RUN wget -O /workspace/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth \
https://huggingface.co/TMElyralab/MuseTalk/resolve/main/dwpose/dw-ll_ucoco_384.pth

# musetalk model
RUN wget -O /workspace/MuseTalk/models/musetalkV15/unet.pth \
https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth

# sd-vae model
RUN wget -O /workspace/MuseTalk/models/sd-vae/diffusion_pytorch_model.safetensors \
https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors

# Back to root
WORKDIR /workspace

# Your app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python", "handler.py"]
