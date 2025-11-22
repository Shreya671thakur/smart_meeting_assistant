import requests
import os
import yake
import numpy as np


# -----------------------------
# Helper: Groq LLM call
# -----------------------------
def groq_chat(prompt, max_tokens=400):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception("Groq LLM error: " + response.text)

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# 1. SUMMARIZATION
# -----------------------------
def summarize_text(text: str, max_length: int = 150):
    prompt = f"""
Summarize the following meeting text in {max_length} words.
Focus on decisions, action items, and key discussion points.

Text:
{text}
"""
    return groq_chat(prompt, max_tokens=300)


# -----------------------------
# 2. KEYPHRASE EXTRACTION
# -----------------------------
def extract_keyphrases(text: str, topk: int = 10):
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=topk)
    keywords = kw_extractor.extract_keywords(text)
    return [k[0] for k in keywords]


# -----------------------------
# 3. SENTIMENT ANALYSIS
# -----------------------------
def get_sentiment(text: str):
    prompt = f"""
Analyze sentiment of this meeting transcript.
Return results in JSON format:
- positive_segments
- negative_segments
- neutral_segments
- overall_sentiment ("positive" / "negative" / "neutral")

Text:
{text}
"""
    result = groq_chat(prompt, max_tokens=300)
    return result  # LLM returns clean JSON or natural text


# -----------------------------
# 4. TOPIC CLUSTERING (LLM-based)
# -----------------------------
def cluster_segments(segment_texts, n_clusters: int = 3):
    prompt = f"""
Cluster the following meeting segments into {n_clusters} topics.
Return output as a JSON dictionary where keys are cluster numbers
and values are lists of segments.

Segments:
{segment_texts}
"""
    result = groq_chat(prompt, max_tokens=400)
    return result  # returns JSON-like clusters


# -----------------------------
# 5. RESUME BULLET GENERATION
# -----------------------------
def generate_resume_bullets(summary_text: str, n: int = 5):
    prompt = f"""
Generate {n} strong, concise resume bullets from this meeting summary.

Requirements:
- Start each bullet with an action verb.
- Use measurable outcomes where possible.
- Keep each bullet one line.
- Professional tone.

Summary:
{summary_text}
"""
    output = groq_chat(prompt, max_tokens=300)

    bullets = [b.strip("-• ").strip() for b in output.split("\n") if b.strip()]
    return bullets[:n]
