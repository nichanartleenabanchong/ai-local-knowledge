"""
Shared logic: load documents, find relevant passages, and ask the local LLM (Ollama).
Used by both app.py (Streamlit chatbot) and evaluate.py (comparison test).
"""
from pathlib import Path

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).parent / "docs"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"   # good with Thai + English. Smaller option: "qwen2.5:3b" or "llama3.2:3b"


# ---------- 1. Load documents and split them into small chunks ----------
def load_chunks(docs_dir=DOCS_DIR):
    """Each paragraph becomes one chunk. The document title is added to every chunk
    so the search knows which festival the paragraph is about."""
    chunks = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = text.strip().splitlines()[0]
        for para in text.split("\n\n"):
            para = para.strip()
            if para and para != title:
                chunks.append({"source": path.name, "text": f"[{title}]\n{para}"})
    return chunks


# ---------- 2. Search: find the chunks most similar to the question ----------
class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        # character n-grams work for both English and Thai (Thai has no spaces between words)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, question, k=4):
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [self.chunks[i] | {"score": float(scores[i])} for i in best]


# ---------- 3. Build prompts for the two modes ----------
SYSTEM_WITH_DOCS = (
    "You are an assistant about Thai festivals and traditions. "
    "Answer ONLY using the information in the CONTEXT below. "
    "If the answer is not in the context, say: \"I don't know based on the documents.\" "
    "Do not make up dates, names or facts. Answer in the same language as the question."
)
SYSTEM_NO_DOCS = (
    "You are an assistant about Thai festivals and traditions. "
    "Answer the question from your own knowledge. Answer in the same language as the question."
)


def build_messages(question, use_docs, retriever=None, k=4):
    """Returns (messages for the LLM, list of retrieved chunks)."""
    if not use_docs:
        return [{"role": "system", "content": SYSTEM_NO_DOCS},
                {"role": "user", "content": question}], []
    hits = retriever.search(question, k=k)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    return [{"role": "system", "content": SYSTEM_WITH_DOCS},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}], hits


# ---------- 4. Call the local LLM through Ollama ----------
def ask_llm(messages, model=DEFAULT_MODEL, temperature=0.2):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False,
              "options": {"temperature": temperature}},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer(question, use_docs, retriever=None, model=DEFAULT_MODEL, k=4):
    messages, hits = build_messages(question, use_docs, retriever, k)
    return ask_llm(messages, model), hits
