import os
import uuid
import subprocess
import requests
import boto3
import shutil
import time
import torchaudio
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

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

s3 = boto3.client("s3", endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY)

print("Loading Chatterbox model...")
t = time.time()
tts_model = ChatterboxTTS.from_pretrained(device='cuda')
print(f"Chatterbox ready in {round(time.time()-t, 1)}s")

def download(url, path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
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

        t1 = time.time()
        print("Downloading avatar...")
        download(avatar_url, avatar_file)
        print(f"⏱ Avatar download: {round(time.time()-t1, 1)}s")

        subprocess.run([
            "ffmpeg", "-y", "-i", str(avatar_file),
            "-t", "10", "-c", "copy", str(avatar_trimmed)
        ], check=True, capture_output=True)

        t2 = time.time()
        print("Downloading voice sample...")
        download(voice_url, voice_raw)
        subprocess.run([
            "ffmpeg", "-y", "-i", str(voice_raw),
            "-ar", "22050", "-ac", "1", str(voice_wav)
        ], check=True, capture_output=True)
        print(f"⏱ Voice download: {round(time.time()-t2, 1)}s")

        t3 = time.time()
        print("Generating speech with Chatterbox...")
        wav = tts_model.generate(
            script,
            audio_prompt_path=str(voice_wav),
            exaggeration=0.5
        )
        torchaudio.save(str(speech_file), wav, tts_model.sr)
        print(f"⏱ Chatterbox TTS: {round(time.time()-t3, 1)}s")

        t4 = time.time()
        print("Running LatentSync...")
        subprocess.run([
            "/venv_chatter/bin/python3",
            "/workspace/LatentSync/scripts/inference.py",
            "--unet_config_path", "/workspace/LatentSync/configs/unet/stage2.yaml",
            "--inference_ckpt_path", "/workspace/LatentSync/checkpoints/latentsync_unet.pt",
            "--inference_steps", "20",
            "--guidance_scale", "1.5",
            "--video_path", str(avatar_trimmed),
            "--audio_path", str(speech_file),
            "--video_out_path", str(output_file)
        ], check=True,
           cwd="/workspace/LatentSync",
           env={**os.environ, "PYTHONPATH": "/workspace/LatentSync"})
        print(f"⏱ LatentSync: {round(time.time()-t4, 1)}s")

        t5 = time.time()
        print("Building final video...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(output_file),
            "-i", str(speech_file),
            "-map", "0:v", "-map", "1:a",
            "-shortest", "-c:v", "copy", "-c:a", "aac",
            str(final_file)
        ], check=True, capture_output=True)

        key = f"generated/{job_id}.mp4"
        s3.upload_file(str(final_file), R2_BUCKET, key)
        url = f"{R2_PUBLIC_BASE}/{key}"
        print(f"⏱ Final + upload: {round(time.time()-t5, 1)}s")
        print(f"⏱ TOTAL: {round(time.time()-t_start, 1)}s")
        print(f"Done: {url}")
        return {"status": "COMPLETED", "video_url": url}

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

import runpod
runpod.serverless.start({"handler": handler})
