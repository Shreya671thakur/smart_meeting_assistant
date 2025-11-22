from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import yake
import numpy as np

# load pipelines lazily
_summarizer = None
_sentiment = None
_embedding_model = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer


def get_sentiment_pipeline():
    global _sentiment
    if _sentiment is None:
        _sentiment = pipeline("sentiment-analysis")
    return _sentiment


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def summarize_text(text: str, max_length: int = 150):
    summarizer = get_summarizer()

    # short text directly
    if len(text) < 1000:
        out = summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        return out[0]['summary_text']

    # long text → chunking
    chunk_size = 900
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = [
        summarizer(c, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
        for c in chunks
    ]

    merged = "\n".join(summaries)
    final = summarizer(
        merged,
        max_length=max_length,
        min_length=30,
        do_sample=False
    )[0]['summary_text']

    return final


def extract_keyphrases(text: str, topk: int = 10):
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=topk)
    keywords = kw_extractor.extract_keywords(text)
    return [k[0] for k in keywords]


def get_sentiment(text: str):
    pipeline_s = get_sentiment_pipeline()

    # split by paragraphs
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paras:
        paras = [text]

    results = pipeline_s(paras[:12])  # limit heavy input

    pos = sum(1 for r in results if r['label'].lower().startswith('pos'))
    neg = sum(1 for r in results if r['label'].lower().startswith('neg'))

    return {
        'positive_segments': pos,
        'negative_segments': neg,
        'detailed': results
    }


def cluster_segments(segment_texts, n_clusters: int = 3):
    model = get_embedding_model()
    emb = model.encode(segment_texts)

    k = min(n_clusters, max(1, len(segment_texts)))
    km = KMeans(n_clusters=k, random_state=42).fit(emb)

    clusters = {i: [] for i in range(k)}
    for idx, label in enumerate(km.labels_):
        clusters[label].append(segment_texts[idx])

    return clusters


def generate_resume_bullets(summary_text: str, n: int = 5):
    generator = pipeline('text2text-generation', model='google/flan-t5-small')

    prompt = (
        f"Extract {n} concise resume bullets from the following meeting summary. "
        f"Use action verbs, measurable outcomes, and professional tone.\n\n"
        f"{summary_text}"
    )

    out = generator(prompt, max_length=250, do_sample=False)[0]['generated_text']

    bullets = [b.strip("-• ").strip() for b in out.split('\n') if b.strip()]
    return bullets[:n]
