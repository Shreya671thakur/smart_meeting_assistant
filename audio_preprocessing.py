from faster_whisper import WhisperModel
from pydub import AudioSegment
import os

model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio_file(path: str):
    audio = AudioSegment.from_file(path)
    wav_path = path.rsplit(".", 1)[0] + "_tmp.wav"
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format='wav')

    segments, info = model.transcribe(wav_path)

    all_text = ""
    normalized_segments = []

    for seg in segments:
        txt = seg.text.strip()
        all_text += txt + " "
        normalized_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": txt
        })

    try:
        os.remove(wav_path)
    except:
        pass

    return all_text.strip(), normalized_segments
