import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import uuid
import subprocess
import requests
import boto3
import shutil
import time
import torchaudio
from pathlib import Path
import runpod

BASE = Path("/workspace/worker")
OUTPUT = BASE / "output"
TMP = BASE / "tmp"
OUTPUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_BUCKET = os.environ["R2_BUCKET"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_PUBLIC_BASE = os.environ["R2_PUBLIC_BASE"]

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
)

tts_model = None

def get_tts():
    global tts_model
    if tts_model is None:
        print("Loading Chatterbox model...")
        from chatterbox.tts import ChatterboxTTS
        t0 = time.time()
        tts_model = ChatterboxTTS.from_pretrained(device="cuda")
        print(f"Chatterbox ready in {round(time.time() - t0, 1)}s")
    return tts_model

def download(url, path):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

def handler(job):
    t_start = time.time()
    job_id = str(uuid.uuid4())
    job_dir = TMP / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        inp = job["input"]
        script = inp["script"]
        avatar_url = inp["avatar_video_url"]
        voice_url = inp["voice_sample_url"]

        avatar_file = job_dir / "avatar.mp4"
        avatar_trimmed = job_dir / "avatar_trimmed.mp4"
        voice_raw = job_dir / "voice_raw"
        voice_wav = job_dir / "voice.wav"
        speech_file = job_dir / "speech.wav"
        output_file = job_dir / "output.mp4"
        final_file = OUTPUT / f"{job_id}.mp4"

        print("Downloading avatar...")
        download(avatar_url, avatar_file)

        subprocess.run([
            "ffmpeg", "-y", "-i", str(avatar_file),
            "-t", "10", "-c", "copy", str(avatar_trimmed)
        ], check=True)

        print("Downloading voice sample...")
        download(voice_url, voice_raw)

        subprocess.run([
            "ffmpeg", "-y", "-i", str(voice_raw),
            "-ar", "22050", "-ac", "1", str(voice_wav)
        ], check=True)

        model = get_tts()

        print("Generating speech...")
        wav = model.generate(
            script,
            audio_prompt_path=str(voice_wav),
            exaggeration=0.5
        )
        torchaudio.save(str(speech_file), wav, model.sr)

        print("Running LatentSync...")
        subprocess.run([
            "python",
            "/workspace/LatentSync/scripts/inference.py",
            "--unet_config_path", "/workspace/LatentSync/configs/unet/stage2.yaml",
            "--inference_ckpt_path", "/workspace/LatentSync/checkpoints/latentsync_unet.pt",
            "--inference_steps", "20",
            "--guidance_scale", "1.5",
            "--video_path", str(avatar_trimmed),
            "--audio_path", str(speech_file),
            "--video_out_path", str(output_file)
        ], check=True, cwd="/workspace/LatentSync", env={**os.environ, "PYTHONPATH": "/workspace/LatentSync"})

        print("Muxing final video...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(output_file),
            "-i", str(speech_file),
            "-map", "0:v", "-map", "1:a",
            "-shortest", "-c:v", "copy", "-c:a", "aac",
            str(final_file)
        ], check=True)

        key = f"generated/{job_id}.mp4"
        s3.upload_file(str(final_file), R2_BUCKET, key)
        url = f"{R2_PUBLIC_BASE}/{key}"

        return {"status": "COMPLETED", "video_url": url}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
