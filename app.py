import streamlit as st
import pandas as pd
import tempfile
from openai import OpenAI

from nlp_utils import (
    summarize_text,
    extract_keyphrases,
    get_sentiment,
    cluster_segments,
    generate_resume_bullets,
)

from utils import save_uploaded_file, seconds_to_hhmmss

# Initialize OpenAI
client = OpenAI()

st.set_page_config(page_title="Smart Meeting Assistant", layout="wide")
st.title("Smart Meeting Assistant — Transcribe, Summarize, Extract")

# Sidebar
with st.sidebar:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader(
        "Upload meeting audio (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a"]
    )

    st.write("Or paste meeting transcript text below:")
    transcript_text = st.text_area(
        "Optional: paste transcript to skip transcription",
        height=150
    )

    run_button = st.button("Process")

# Helper: Transcribing via OpenAI API
def transcribe_audio_file(audio_path):
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    text = transcript.text

    # API does not provide segments → we create dummy segments
    segs = [{"start": 0, "end": 0, "text": p} for p in text.split(". ") if p.strip()]
    return text, segs


# Main Logic
if run_button:

    if not uploaded_file and not transcript_text:
        st.warning("Upload audio or paste transcript to continue")

    else:
        # Transcribe audio
        if uploaded_file:
            path = save_uploaded_file(uploaded_file)
            with st.spinner("Transcribing audio..."):
                text, segments = transcribe_audio_file(path)

        else:
            text = transcript_text

            # Fake segments for analytics
            seg_texts = [p.strip() for p in text.split('\n\n') if p.strip()]
            segments = [
                {'start': 0, 'end': 0, 'text': s}
                for s in seg_texts
            ]

        # Show transcript
        st.subheader("Full Transcript")
        st.write(text[:10000])

        # Summaries
        st.subheader("Summaries")
        short_sum = summarize_text(text, max_length=120)
        long_sum = summarize_text(text, max_length=400)

        st.markdown("**Short summary:**")
        st.write(short_sum)

        st.markdown("**Long summary:**")
        st.write(long_sum)

        # Keyphrases
        st.subheader("Keyphrases & Action Items")
        keyphrases = extract_keyphrases(text, topk=15)
        st.write(keyphrases)

        # Sentiment
        st.subheader("Sentiment")
        sentiment_overall = get_sentiment(text)
        st.write(sentiment_overall)

        # Clusters
        st.subheader("Topic Clusters")
        clusters = cluster_segments([s['text'] for s in segments])
        st.write(clusters)

        # Resume Bullets
        st.subheader("Generate Resume Bullets")
        bullets = generate_resume_bullets(short_sum, n=5)
        for b in bullets:
            st.write("- " + b)

        st.success("Processing complete!")
