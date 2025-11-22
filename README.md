#click here to view my app
https://smartmeetingassistant-pmfy6brtymsy6jwpfhwqcy.streamlit.app/

# example: use vosk
from vosk import Model, KaldiRecognizer
import wave

model = Model('model')  # folder path to vosk model
wf = wave.open('file.wav', 'rb')

rec = KaldiRecognizer(model, wf.getframerate())
results = []

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break

    if rec.AcceptWaveform(data):
        results.append(rec.Result())

# after loop ends
results.append(rec.FinalResult())

