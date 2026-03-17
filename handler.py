import runpod
import uuid
import subprocess
import requests
import boto3
import shutil
from pathlib import Path

BASE = Path("/workspace/worker")
DOWNLOADS = BASE / "downloads"
OUTPUT = BASE / "output"
TMP = BASE / "tmp"

for p in [DOWNLOADS, OUTPUT, TMP]:
    p.mkdir(parents=True, exist_ok=True)

R2_ENDPOINT = "https://f4c75ecb147945e4591fead5f2002e92.r2.cloudflarestorage.com"
R2_BUCKET = "intooya-videos"
R2_ACCESS_KEY = "b93115b519c5c707136452918194ed42"
R2_SECRET_KEY = "d5986eb86c0bd3525bca6dadfca1e21b8e000d00a87c08382c5acd04e4e387d6"
R2_PUBLIC_BASE = "https://pub-fd10b60f6b9c468fb83b93b7625f32a9.r2.dev"

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
)

def run(cmd):
    print("RUN:", cmd)
    subprocess.run(cmd, check=True)

def download(url, path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk)

def handler(job):

    job_id = str(uuid.uuid4())
    job_dir = TMP / job_id
    job_dir.mkdir()

    try:
        inp = job["input"]

        script = inp["script"]
        avatar_url = inp["avatar_video_url"]
        voice_url = inp["voice_sample_url"]

        avatar_file = job_dir / "avatar.mp4"
        voice_file = job_dir / "voice.wav"
        speech_file = job_dir / "speech.wav"

        print("Downloading avatar...")
        download(avatar_url, avatar_file)

        print("Downloading voice...")
        download(voice_url, voice_file)

        print("Generating speech (XTTS)...")
        run([
            "/workspace/venvs/xtts/bin/python",
            "/workspace/xtts_infer.py",
            script,
            str(voice_file),
            str(speech_file)
        ])

        print("Running MuseTalk...")

        frames = job_dir / "frames"
        frames.mkdir()

        run([
            "bash",
            "-c",
            f"cd /workspace/MuseTalk && /workspace/venvs/musetalk/bin/python scripts/inference.py \
            --inference_config configs/inference/realtime.yaml \
            --result_dir {frames} \
            --version v15 \
            --video_path {avatar_file} \
            --audio_path {speech_file}"
        ])

        final_video = OUTPUT / f"{job_id}.mp4"

        print("Encoding video...")
        run([
            "ffmpeg","-y",
            "-framerate","25",
            "-i",str(frames / "%08d.png"),
            "-i",str(speech_file),
            "-c:v","libx264",
            "-pix_fmt","yuv420p",
            "-shortest",
            str(final_video)
        ])

        print("Uploading to R2...")
        key = f"generated/{job_id}.mp4"
        s3.upload_file(str(final_video), R2_BUCKET, key)

        url = f"{R2_PUBLIC_BASE}/{key}"

        return {
            "status": "COMPLETED",
            "video_url": url
        }

    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e)
        }

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
