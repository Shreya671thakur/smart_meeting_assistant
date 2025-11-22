import streamlit as st
import pandas as pd
import requests
import os

from nlp_utils import (
    summarize_text,
    extract_keyphrases,
    get_sentiment,
    cluster_segments,
    generate_resume_bullets,
)

from utils import save_uploaded_file, seconds_to_hhmmss


st.set_page_config(page_title="Smart Meeting Assistant", layout="wide")
st.title("Smart Meeting Assistant — Transcribe, Summarize, Extract")


# -------------------------
#  GROQ Whisper Transcriber
# -------------------------
def transcribe_with_groq(audio_path):

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}

    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"model": "whisper-large-v3"}

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        st.error("Groq API Error: " + response.text)
        return "", []

    text = response.json().get("text", "")

    segments = [{"start": 0, "end": 0, "text": p} 
                for p in text.split(". ") if p.strip()]

    return text, segments


# -------------------------
#        SIDEBAR
# -------------------------
with st.sidebar:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader(
        "Upload meeting audio (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a"]
    )

    st.write("Or paste transcript text below:")
    transcript_text = st.text_area(
        "Optional: paste transcript",
        height=150
    )

    run_button = st.button("Process")


# -------------------------
#        MAIN LOGIC
# -------------------------
if run_button:

    if not uploaded_file and not transcript_text:
        st.warning("Upload audio or paste transcript to continue.")

    else:
        # ================
        # Step 1: Transcript
        # ================
        if uploaded_file:
            path = save_uploaded_file(uploaded_file)

            with st.spinner("Transcribing audio via Whisper (Groq)..."):
                text, segments = transcribe_with_groq(path)

        else:
            text = transcript_text
            segments = [{"start": 0, "end": 0, "text": s}
                        for s in text.split("\n\n") if s.strip()]

        # ================
        # Show transcript
        # ================
        st.subheader("Full Transcript")
        st.write(text)

        # ================
        # Step 2: Summaries
        # ================
        st.subheader("Summaries")

        short_sum = summarize_text(text, max_length=120)
        long_sum = summarize_text(text, max_length=400)

        st.markdown("**Short Summary:**")
        st.write(short_sum)

        st.markdown("**Long Summary:**")
        st.write(long_sum)

        # ================
        # Step 3: Keyphrases
        # ================
        st.subheader("Keyphrases & Action Items")
        st.write(extract_keyphrases(text, topk=15))

        # ================
        # Step 4: Sentiment
        # ================
        st.subheader("Sentiment")
        st.write(get_sentiment(text))

        # ================
        # Step 5: Topic Clusters
        # ================
        st.subheader("Topic Clusters")
        st.write(cluster_segments([s["text"] for s in segments]))

        # ================
        # Step 6: Resume Bullets
        # ================
        st.subheader("Generate Resume Bullets")
        bullets = generate_resume_bullets(short_sum, n=5)

        for b in bullets:
            st.write("• " + b)

        st.success("Processing complete!")



