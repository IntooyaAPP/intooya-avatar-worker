FROM python:3.10

# Install system deps
RUN apt-get update && apt-get install -y ffmpeg git

WORKDIR /workspace

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git

# Install MuseTalk dependencies
WORKDIR /workspace/MuseTalk
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .
# Go back to root
WORKDIR /workspace

# Install your app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python", "handler.py"]
