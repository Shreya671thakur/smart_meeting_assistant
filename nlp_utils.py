from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import yake

_summarizer = None
_sentiment = None
_embedding = None
_generator = None


def summarize_text(text: str, max_length=150):
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    chunks = [text[i:i + 700] for i in range(0, len(text), 700)]

    summaries = [
        _summarizer(c, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
        for c in chunks
    ]

    final = " ".join(summaries)
    return final[:max_length * 5]  # safe max length


def extract_keyphrases(text, topk=10):
    kw = yake.KeywordExtractor(lan="en", n=3, top=topk)
    return [k[0] for k in kw.extract_keywords(text)]


def get_sentiment(text):
    global _sentiment
    if _sentiment is None:
        _sentiment = pipeline("sentiment-analysis")

    paras = [p.strip() for p in text.split("\n") if p.strip()][:12]
    res = _sentiment(paras)

    pos = sum(1 for r in res if r['label'] == 'POSITIVE')
    neg = sum(1 for r in res if r['label'] == 'NEGATIVE')

    return {"positive": pos, "negative": neg, "detailed": res}


def cluster_segments(segments, n_clusters=3):
    global _embedding
    if _embedding is None:
        _embedding = SentenceTransformer("all-MiniLM-L6-v2")

    emb = _embedding.encode(segments)

    k = min(n_clusters, len(segments))
    km = KMeans(n_clusters=k, random_state=42).fit(emb)

    clusters = {i: [] for i in range(k)}
    for idx, label in enumerate(km.labels_):
        clusters[label].append(segments[idx])

    return clusters


def generate_resume_bullets(summary, n=5):
    global _generator
    if _generator is None:
        _generator = pipeline("text2text-generation", model="google/flan-t5-small")

    prompt = f"Generate {n} resume bullets from this summary:\n{summary}"

    out = _generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]

    bullets = [b.strip("-• ").strip() for b in out.split("\n") if b.strip()]
    return bullets[:n]
