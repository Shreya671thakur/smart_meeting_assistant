import os
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def save_uploaded_file(uploaded_file) -> str:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def seconds_to_hhmmss(s: float) -> str:
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"
