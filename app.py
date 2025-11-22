import streamlit as st
import pandas as pd

from audio_preprocessing import transcribe_audio_file
from nlp_utils import (
    summarize_text,
    extract_keyphrases,
    get_sentiment,
    cluster_segments,
    generate_resume_bullets,
)
from utils import save_uploaded_file

st.set_page_config(page_title="Smart Meeting Assistant", layout="wide")
st.title("Smart Meeting Assistant — Free Offline Version")

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload meeting audio (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a"]
    )

    transcript_text = st.text_area(
        "Or paste transcript text:",
        height=150
    )

    run_button = st.button("Process")


if run_button:
    if not uploaded_file and not transcript_text:
        st.warning("Upload an audio file or paste transcript text.")
    else:
        # Transcription
        if uploaded_file:
            path = save_uploaded_file(uploaded_file)
            st.info("Transcribing audio (local Faster-Whisper)...")
            text, segments = transcribe_audio_file(path)
        else:
            text = transcript_text
            segments = [{'start': 0, 'end': 0, 'text': t} for t in text.split(". ")]

        # Show transcript
        st.subheader("Transcript")
        st.write(text)

        # Summaries
        st.subheader("Summaries")
        st.write("Generating summary...")
        short = summarize_text(text, max_length=120)
        long = summarize_text(text, max_length=300)
        st.write("**Short summary:**")
        st.write(short)
        st.write("**Long summary:**")
        st.write(long)

        # Keyphrases
        st.subheader("Keyphrases")
        st.write(extract_keyphrases(text, topk=15))

        # Sentiment
        st.subheader("Sentiment")
        st.write(get_sentiment(text))

        # Clusters
        st.subheader("Topic Clusters")
        st.write(cluster_segments([s['text'] for s in segments]))

        # Resume Bullets
        st.subheader("Resume Bullets")
        bullets = generate_resume_bullets(short, n=5)
        for b in bullets:
            st.write("- " + b)

        st.success("Processing complete!")
