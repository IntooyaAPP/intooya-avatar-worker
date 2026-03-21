import os
import uuid
import subprocess
import requests
import boto3
import shutil
import hashlib
import glob
import time
from pathlib import Path

BASE = Path("/runpod-volume/worker")
OUTPUT = BASE / "output"
TMP = BASE / "tmp"
OUTPUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

R2_ENDPOINT = "https://f4c75ecb147945e4591fead5f2002e92.r2.cloudflarestorage.com"
R2_BUCKET = "intooya-videos"
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_PUBLIC_BASE = "https://pub-fd10b60f6b9c468fb83b93b7625f32a9.r2.dev"

s3 = boto3.client("s3", endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY)

os.environ["COQUI_TOS_AGREED"] = "1"
print("Handler starting up.")

def download(url, path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)

def avatar_cache_key(avatar_url):
    url_hash = hashlib.md5(avatar_url.encode()).hexdigest()
    return f"avatar_cache_v2/{url_hash}"

def upload_dir_to_r2(local_dir, r2_prefix):
    for root, dirs, files in os.walk(local_dir):
        dirs[:] = [d for d in dirs if d != 'full_imgs']
        for file in files:
            local_path = os.path.join(root, file)
            relative = os.path.relpath(local_path, local_dir)
            r2_key = f"{r2_prefix}/{relative}"
            s3.upload_file(local_path, R2_BUCKET, r2_key)
    print(f"Uploaded avatar cache to R2: {r2_prefix}")

def download_dir_from_r2(r2_prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=R2_BUCKET, Prefix=r2_prefix + "/")
    found = False
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(r2_prefix)+1:]
            if not relative:
                continue
            local_path = os.path.join(local_dir, relative)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(R2_BUCKET, key, local_path)
            found = True
    return found

def handler(job):
    t_start = time.time()
    job_id = str(uuid.uuid4())
    job_dir = TMP / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    results_dir = f"/runpod-volume/results/{job_id}"
    yaml_path = f"/runpod-volume/MuseTalk/configs/inference/realtime_{job_id}.yaml"

    try:
        inp = job["input"]
        script = inp["script"]
        avatar_url = inp["avatar_video_url"]
        voice_url = inp["voice_sample_url"]
        r2_cache_prefix = avatar_cache_key(avatar_url)
        avatar_id = hashlib.md5(avatar_url.encode()).hexdigest()[:8]
        avatar_cache_dir = f"/runpod-volume/avatars/{avatar_id}"
        avatar_file = job_dir / "avatar.mp4"
        voice_file = job_dir / "voice.wav"
        speech_file = job_dir / "speech.wav"

        t1 = time.time()
        print("Downloading avatar...")
        download(avatar_url, avatar_file)
        print(f"⏱ Avatar download: {round(time.time()-t1, 1)}s")

        t2 = time.time()
        print("Downloading voice sample...")
        voice_raw = job_dir / "voice_raw"
        download(voice_url, voice_raw)
        subprocess.run(["/runpod-volume/ffmpeg", "-y", "-i", str(voice_raw),
            "-ar", "22050", "-ac", "1", str(voice_file)], check=True)
        print(f"⏱ Voice download + convert: {round(time.time()-t2, 1)}s")

        t3 = time.time()
        print("Checking avatar cache...")
        has_cache = download_dir_from_r2(r2_cache_prefix, avatar_cache_dir)
        preparation = "false" if has_cache else "true"
        if has_cache:
            print(f"⏱ Cache download: {round(time.time()-t3, 1)}s")
            print("Using cached avatar - fast inference!")
            musetalk_avatar_dir = f"/runpod-volume/MuseTalk/results/v15/avatars/{avatar_id}"
            if not os.path.exists(musetalk_avatar_dir):
                t3b = time.time()
                shutil.copytree(avatar_cache_dir, musetalk_avatar_dir)
                print(f"⏱ Cache restore to MuseTalk: {round(time.time()-t3b, 1)}s")
                print(f"Cache restored to {musetalk_avatar_dir}")
        else:
            print(f"⏱ Cache check (miss): {round(time.time()-t3, 1)}s")
            print("First time - preprocessing avatar (one-time cost)...")

        t4 = time.time()
        print("Generating speech with XTTS...")
        subprocess.run([
            "/runpod-volume/venvs/xtts/bin/python",
            "/runpod-volume/xtts_infer.py",
            script, str(voice_file), str(speech_file)
        ], check=True)
        print(f"⏱ XTTS speech generation: {round(time.time()-t4, 1)}s")

        os.makedirs(results_dir, exist_ok=True)

        yaml_content = (
            f"{avatar_id}:\n"
            f"  preparation: {preparation}\n"
            f"  video_path: \"{avatar_file}\"\n"
            "  bbox_shift: 5\n"
            "  audio_clips:\n"
            f"    audio_0: \"{speech_file}\"\n"
        )
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        t5 = time.time()
        cmd = [
            "/runpod-volume/venvs/musetalk/bin/python",
            "scripts/realtime_inference.py",
            "--inference_config", yaml_path,
            "--result_dir", results_dir,
            "--version", "v15",
            "--ffmpeg_path", "/runpod-volume",
            "--batch_size", "20",
            "--unet_config", "./models/musetalkV15/musetalk.json",
            "--unet_model_path", "./models/musetalkV15/unet.pth",
        ]
        subprocess.run(cmd, check=True, cwd="/runpod-volume/MuseTalk",
            env={**os.environ, "PYTHONPATH": "/runpod-volume/MuseTalk"})
        print(f"⏱ MuseTalk inference: {round(time.time()-t5, 1)}s")

        if not has_cache:
            t5b = time.time()
            avatar_result_dir = f"/runpod-volume/MuseTalk/results/v15/avatars/{avatar_id}"
            if os.path.exists(avatar_result_dir):
                upload_dir_to_r2(avatar_result_dir, r2_cache_prefix)
                print("Avatar cache uploaded successfully")
            else:
                print(f"WARNING: Cache dir not found at {avatar_result_dir}")
            print(f"⏱ Cache upload: {round(time.time()-t5b, 1)}s")

        vids = glob.glob(f"{results_dir}/**/*.mp4", recursive=True)
        if not vids:
            vids = glob.glob(f"/runpod-volume/MuseTalk/results/v15/avatars/{avatar_id}/vid_output/*.mp4")
        if not vids:
            raise Exception("No output video found")
        output_vid = vids[0]

        t6 = time.time()
        print("Building final video...")
        final_video = OUTPUT / f"{job_id}.mp4"
        subprocess.run([
            "/runpod-volume/ffmpeg", "-y",
            "-i", output_vid,
            "-i", str(speech_file),
            "-map", "0:v", "-map", "1:a",
            "-shortest", "-c:v", "copy", "-c:a", "aac",
            str(final_video)
        ], check=True)

        key = f"generated/{job_id}.mp4"
        s3.upload_file(str(final_video), R2_BUCKET, key)
        url = f"{R2_PUBLIC_BASE}/{key}"
        print(f"⏱ Final video + upload: {round(time.time()-t6, 1)}s")
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
        shutil.rmtree(avatar_cache_dir, ignore_errors=True)
        shutil.rmtree(results_dir, ignore_errors=True)
        Path(yaml_path).unlink(missing_ok=True)

import runpod
runpod.serverless.start({"handler": handler})
