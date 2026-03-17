FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace/MuseTalk"

RUN apt-get update && apt-get install -y ffmpeg git

WORKDIR /workspace

# Prebuilt torch (fast)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git

WORKDIR /workspace/MuseTalk

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "handler.py"]
