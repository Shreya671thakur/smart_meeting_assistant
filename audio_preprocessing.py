import whisper
import os
import subprocess

model = whisper.load_model("base")

def transcribe_audio_file(file_path):
    # Ensure ffmpeg can read the file
    output_wav = "temp.wav"
    subprocess.run([
        "ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", output_wav, "-y"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    result = model.transcribe(output_wav)
    os.remove(output_wav)
    return result["text"]
