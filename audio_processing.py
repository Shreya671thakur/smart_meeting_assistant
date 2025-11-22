import requests
import os
from pydub import AudioSegment
import tempfile


def transcribe_audio_file(path: str):
    """
    Transcribes an audio file using Groq Whisper API.
    Returns:
        text (str), segments (list of dict with start, end, text)
    """

    # Convert to WAV 16k (Groq Whisper prefers clean audio)
    audio = AudioSegment.from_file(path)
    wav_path = path.rsplit('.', 1)[0] + "_converted.wav"
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")

    # Groq API endpoint
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
    }

    with open(wav_path, "rb") as f:
        files = {"file": f}
        data = {"model": "whisper-large-v3"}

        response = requests.post(url, headers=headers, files=files, data=data)

    # Cleanup
    try:
        os.remove(wav_path)
    except:
        pass

    if response.status_code != 200:
        raise Exception("Groq API Error: " + response.text)

    resp = response.json()
    text = resp.get("text", "")

    # Groq Whisper does NOT return segments → create simple pseudo segments
    segments = [
        {"start": 0, "end": 0, "text": p}
        for p in text.split(". ") if p.strip()
    ]

    return text, segments
