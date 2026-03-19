import sys
from TTS.api import TTS

if len(sys.argv) != 4:
    print("usage: python xtts_infer.py <text> <speaker_wav> <output_wav>")
    sys.exit(1)

text = sys.argv[1]
speaker_wav = sys.argv[2]
output_wav = sys.argv[3]

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text=text,
    speaker_wav=speaker_wav,
    language="en",
    file_path=output_wav
)

print(output_wav)
