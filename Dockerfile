FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace/MuseTalk"

RUN apt-get update && apt-get install -y ffmpeg git curl

WORKDIR /workspace

# Torch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Stable OpenMMLab (no mim)
RUN pip install --no-cache-dir mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
RUN pip install --no-cache-dir mmpose==1.2.0

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git

WORKDIR /workspace/MuseTalk
RUN pip install --no-cache-dir -r requirements.txt

# Create model dirs
RUN mkdir -p /workspace/MuseTalk/models/dwpose
RUN mkdir -p /workspace/MuseTalk/models/musetalkV15
RUN mkdir -p /workspace/MuseTalk/models/sd-vae

# Download models (FIXED)
RUN curl -L https://huggingface.co/TMElyralab/MuseTalk/resolve/main/dwpose/dw-ll_ucoco_384.pth \
-o /workspace/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth

RUN curl -L https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth \
-o /workspace/MuseTalk/models/musetalkV15/unet.pth

RUN curl -L https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors \
-o /workspace/MuseTalk/models/sd-vae/diffusion_pytorch_model.safetensors

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "handler.py"]
