FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace/MuseTalk"

RUN apt-get update && apt-get install -y ffmpeg git curl

WORKDIR /workspace

# Clone MuseTalk first
RUN git clone https://github.com/TMElyralab/MuseTalk.git

WORKDIR /workspace/MuseTalk

# Official MuseTalk install flow
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --editable ./musetalk/whisper
RUN pip install --no-cache-dir -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.1"
RUN mim install "mmdet>=3.1.0"
RUN mim install "mmpose>=1.1.0"

# Model dirs
RUN mkdir -p /workspace/MuseTalk/models/dwpose
RUN mkdir -p /workspace/MuseTalk/models/musetalkV15
RUN mkdir -p /workspace/MuseTalk/models/sd-vae

# Model downloads
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
COPY handler.py .

CMD ["python", "handler.py"]
