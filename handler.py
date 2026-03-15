import runpod

def handler(job):
    job_input = job["input"]

    script = job_input.get("script")
    avatar_video_url = job_input.get("avatar_video_url")
    voice_sample_url = job_input.get("voice_sample_url")

    print("Received job:")
    print(script)
    print(avatar_video_url)
    print(voice_sample_url)

    # temporary test response
    return {
        "status": "worker running",
        "script": script
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
