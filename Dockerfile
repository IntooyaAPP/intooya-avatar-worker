FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/MuseTalk
ENV COQUI_TOS_AGREED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg git wget curl espeak-ng python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git /workspace/MuseTalk

# Create venvs
RUN python3 -m venv /workspace/venvs/musetalk
RUN python3 -m venv /workspace/venvs/xtts

# Install musetalk dependencies
COPY musetalk_requirements.txt /workspace/musetalk_requirements.txt
RUN /workspace/venvs/musetalk/bin/pip install --upgrade pip && \
    /workspace/venvs/musetalk/bin/pip install \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r /workspace/musetalk_requirements.txt

# Install xtts dependencies
COPY xtts_requirements.txt /workspace/xtts_requirements.txt
RUN /workspace/venvs/xtts/bin/pip install --upgrade pip && \
    /workspace/venvs/xtts/bin/pip install -r /workspace/xtts_requirements.txt

# Install runpod in both venvs
RUN /workspace/venvs/musetalk/bin/pip install runpod boto3 requests
RUN /workspace/venvs/xtts/bin/pip install runpod boto3 requests

# Download MuseTalk models
RUN mkdir -p /workspace/MuseTalk/models/musetalkV15 && \
    wget -q -O /workspace/MuseTalk/models/musetalkV15/unet.pth \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth"

RUN mkdir -p /workspace/MuseTalk/models/musetalk && \
    wget -q -O /workspace/MuseTalk/models/musetalk/config.json \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/config.json"

# v1.5 config with 384 dims (matches actual weights)
RUN echo '{"_class_name":"UNet2DConditionModel","_diffusers_version":"0.6.0.dev0","act_fn":"silu","attention_head_dim":8,"block_out_channels":[320,640,1280,1280],"center_input_sample":false,"cross_attention_dim":384,"down_block_types":["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],"downsample_padding":1,"flip_sin_to_cos":true,"freq_shift":0,"in_channels":8,"layers_per_block":2,"mid_block_scale_factor":1,"norm_eps":1e-05,"norm_num_groups":32,"out_channels":4,"sample_size":64,"up_block_types":["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"]}' \
    > /workspace/MuseTalk/models/musetalkV15/musetalk.json

RUN mkdir -p /workspace/MuseTalk/models/face-parse-bisent && \
    wget -q -O /workspace/MuseTalk/models/face-parse-bisent/79999_iter.pth "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth" && \
    wget -q -O /workspace/MuseTalk/models/face-parse-bisent/resnet18-5c106cde.pth "https://download.pytorch.org/models/resnet18-5c106cde.pth"

RUN mkdir -p /workspace/MuseTalk/models/dwpose && \
    wget -q -O /workspace/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth \
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth"

RUN mkdir -p /workspace/MuseTalk/models/face-parse-bisent && \
    wget -q -O /workspace/MuseTalk/models/face-parse-bisent/79999_iter.pth \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth" && \
    wget -q -O /workspace/MuseTalk/models/face-parse-bisent/resnet18-5
    "https://download.pytorch.org/models/resnet18-5c106cde.pth"

# Whisper tiny (384 dims — matches UNet)
RUN mkdir -p /workspace/MuseTalk/models/whisper && \
    wget -q -O /workspace/MuseTalk/models/whisper/config.json \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json" && \
    wget -q -O /workspace/MuseTalk/models/whisper/pytorch_model.bin \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin" && \
    wget -q -O /workspace/MuseTalk/models/whisper/preprocessor_config.json \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json" && \
    wget -q -O /workspace/MuseTalk/models/whisper/tokenizer_config.json \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer_config.json" && \
    wget -q -O /workspace/MuseTalk/models/whisper/vocab.json \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json" && \
    wget -q -O /workspace/MuseTalk/models/whisper/merges.txt \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/merges.txt" && \
    wget -q -O /workspace/MuseTalk/models/whisper/special_tokens_map.json \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/special_tokens_map.json"

# Download XTTS models at build time
RUN /workspace/venvs/xtts/bin/python -c \
    "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Fix MuseTalk inference.py
RUN sed -i 's|default="./models/musetalk/config.json"|default="./models/musetalkV15/musetalk.json"|' \
    /workspace/MuseTalk/scripts/inference.py

# Copy working files
COPY realtime.yaml /workspace/MuseTalk/configs/inference/realtime.yaml
COPY handler.py /workspace/handler.py
COPY xtts_infer.py /workspace/xtts_infer.py

# Create required directories
RUN mkdir -p /workspace/worker/output /workspace/worker/tmp /workspace/test /workspace/results

CMD ["python", "/workspace/handler.py"]
