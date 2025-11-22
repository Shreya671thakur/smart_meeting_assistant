import whisper
from pydub import AudioSegment
import os

model = whisper.load_model("base")  
# change to small/medium for better accuracy if you have RAM

def transcribe_audio_file(path: str):
    """
    Transcribes an audio file and returns full text and segments.
    Returns:
        text (str), segments (list of dict with start, end, text)
    """
    
    # convert to wav 16k if needed
    audio = AudioSegment.from_file(path)
    wav_path = path.rsplit('.', 1)[0] + "_converted.wav"
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format='wav')

    result = model.transcribe(wav_path, verbose=False)
    text = result.get('text', '')

    # Whisper returns segments with timestamps
    segments = result.get('segments', [])

    # normalize segments to our preferred format
    norm_segments = []
    for seg in segments:
        norm_segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text']
        })

    # cleanup temp file
    try:
        os.remove(wav_path)
    except Exception:
        pass

    return text, norm_segments
