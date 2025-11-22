import whisper
import os
import subprocess

model = whisper.load_model("base")

def transcribe_audio_file(file_path):
    output_wav = "temp.wav"

    subprocess.run([
        "ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", output_wav, "-y"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    result = model.transcribe(output_wav, verbose=False)

    os.remove(output_wav)

    return result["text"], result["segments"]
