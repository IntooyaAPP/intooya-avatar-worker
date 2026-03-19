import os
import uuid
import subprocess
import requests
import boto3
import shutil
import hashlib
from pathlib import Path

BASE = Path("/workspace/worker")
OUTPUT = BASE / "output"
TMP = BASE / "tmp"

OUTPUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

TEST_DIR = Path("/workspace/test")
TEST_DIR.mkdir(parents=True, exist_ok=True)

R2_ENDPOINT = "https://f4c75ecb147945e4591fead5f2002e92.r2.cloudflarestorage.com"
R2_BUCKET = "intooya-videos"
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
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
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)

def avatar_cache_key(avatar_url):
    url_hash = hashlib.md5(avatar_url.encode()).hexdigest()
    return f"avatar_cache/{url_hash}.pkl"

def download_cached_coords(avatar_url, dest_path):
    key = avatar_cache_key(avatar_url)
    try:
        s3.download_file(R2_BUCKET, key, str(dest_path))
        print(f"✅ Found cached avatar coords: {key}")
        return True
    except Exception:
        print(f"No cached coords found for this avatar — will extract fresh.")
        return False

def upload_cached_coords(avatar_url, pkl_path):
    key = avatar_cache_key(avatar_url)
    s3.upload_file(str(pkl_path), R2_BUCKET, key)
    print(f"✅ Cached avatar coords uploaded: {key}")

def handler(job):
    job_id = str(uuid.uuid4())
    job_dir = TMP / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

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

        print("Downloading voice sample...")
        download(voice_url, voice_file)

        print("Generating speech with XTTS...")
        run([
            "/workspace/venvs/xtts/bin/python",
            "/workspace/xtts_infer.py",
            script,
            str(voice_file),
            str(speech_file)
        ])

        print("Copying files into working directory...")
        shutil.copy(avatar_file, TEST_DIR / "video.mp4")
        shutil.copy(speech_file, TEST_DIR / "audio.wav")

        print("Resetting results directory...")
        shutil.rmtree("/workspace/results", ignore_errors=True)
        Path("/workspace/results").mkdir(parents=True, exist_ok=True)

        pkl_path = Path("/workspace/video.pkl")
        has_cache = download_cached_coords(avatar_url, pkl_path)

        cmd = [
            "/workspace/venvs/musetalk/bin/python",
            "scripts/inference.py",
            "--inference_config", "configs/inference/realtime.yaml",
            "--result_dir", "/workspace/results",
            "--version", "v15",
            "--saved_coord",
        ]

        if has_cache:
            cmd.append("--use_saved_coord")
            print("🚀 Using cached avatar coords — skipping landmark extraction!")
        else:
            print("⏳ First time for this avatar — extracting landmarks...")

        subprocess.run(
            cmd,
            check=True,
            cwd="/workspace/MuseTalk",
            env={**os.environ, "PYTHONPATH": "/workspace/MuseTalk"},
        )

        if not has_cache and pkl_path.exists():
            upload_cached_coords(avatar_url, pkl_path)

        print("Building final video...")
        final_video = OUTPUT / f"{job_id}.mp4"
        shutil.copy("/workspace/results/v15/video_audio.mp4", str(final_video))

        key = f"generated/{job_id}.mp4"
        s3.upload_file(str(final_video), R2_BUCKET, key)

        url = f"{R2_PUBLIC_BASE}/{key}"
        print(f"✅ Done: {url}")
        return {"status": "COMPLETED", "video_url": url}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "FAILED", "error": str(e)}

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

import runpod
runpod.serverless.start({"handler": handler})
