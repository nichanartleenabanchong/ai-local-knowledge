"""
Thai Festivals Chatbot (Songkran, Loy Krathong, ...)
Compare an AI answering WITH local documents (RAG) vs WITHOUT them.

Run locally:   streamlit run app.py
API key:       .streamlit/secrets.toml  ->  ANTHROPIC_API_KEY = "..."
               or environment variable ANTHROPIC_API_KEY
"""
import os
from pathlib import Path

import anthropic
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).parent / "docs"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
CHUNK_CHARS = 700

MODE_RAG = "With documents (RAG)"
MODE_BASE = "Without documents (baseline)"
MODE_BOTH = "Compare both side by side"


# ---------- knowledge base ----------
def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS):
    """Split on blank lines, then merge small pieces up to max_chars."""
    chunks, buf = [], ""
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if buf and len(buf) + len(para) > max_chars:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


@st.cache_resource
def load_index():
    chunks, sources = [], []
    for f in sorted(DOCS_DIR.glob("*")):
        if f.suffix.lower() not in {".md", ".txt"}:
            continue
        for c in split_into_chunks(f.read_text(encoding="utf-8")):
            chunks.append(c)
            sources.append(f.name)
    if not chunks:
        return None
    # character n-grams work for both Thai (no spaces) and English
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    matrix = vec.fit_transform(chunks)
    return {"chunks": chunks, "sources": sources, "vec": vec, "matrix": matrix}


def retrieve(index, question: str, k: int):
    q = index["vec"].transform([question])
    sims = cosine_similarity(q, index["matrix"]).ravel()
    top = sims.argsort()[::-1][:k]
    return [(index["sources"][i], index["chunks"][i], float(sims[i])) for i in top]


# ---------- model calls ----------
def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            key = None
    if not key:
        st.error("Missing ANTHROPIC_API_KEY. Add it to Streamlit secrets or set it as an environment variable.")
        st.stop()
    return anthropic.Anthropic(api_key=key)


def ask_baseline(client, question: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=600,
        temperature=0,
        system="You are a helpful assistant. Answer the question concisely in the same language as the question.",
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text


def ask_rag(client, question: str, hits) -> str:
    context = "\n\n---\n\n".join(f"[{src}]\n{text}" for src, text, _ in hits)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=600,
        temperature=0,
        system=(
            "Answer ONLY using the documents provided. "
            "If the answer is not in the documents, say: 'Not found in the documents.' "
            "Do not use outside knowledge. Answer concisely in the same language as the question."
        ),
        messages=[{"role": "user", "content": f"DOCUMENTS:\n{context}\n\nQUESTION: {question}"}],
    )
    return resp.content[0].text


# ---------- UI ----------
st.set_page_config(page_title="Thai Festivals AI", page_icon="🪔")
st.title("🪔 Thai Festivals AI")
st.caption("Does giving an AI local documents make it more accurate? Songkran, Loy Krathong and more.")

index = load_index()
if index is None:
    st.error("No .md or .txt files found in the docs/ folder.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Answer mode", [MODE_RAG, MODE_BASE, MODE_BOTH])
    k = st.slider("Chunks retrieved (k)", 1, 8, 4)
    st.markdown(f"**Documents loaded:** {len(set(index['sources']))} files, {len(index['chunks'])} chunks")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        if item["mode"] == MODE_BOTH:
            c1, c2 = st.columns(2)
            c1.markdown("**With documents**")
            c1.write(item["rag"])
            c2.markdown("**Without documents**")
            c2.write(item["base"])
        else:
            st.write(item["rag"] if item["mode"] == MODE_RAG else item["base"])
        if item.get("hits"):
            with st.expander("Retrieved sources"):
                for src, text, score in item["hits"]:
                    st.markdown(f"**{src}** (score {score:.2f})")
                    st.caption(text)

question = st.chat_input("Ask about Songkran, Loy Krathong, dates, rituals, regional differences...")
if question:
    client = get_client()
    entry = {"question": question, "mode": mode, "rag": None, "base": None, "hits": None}
    with st.spinner("Thinking..."):
        if mode in (MODE_RAG, MODE_BOTH):
            entry["hits"] = retrieve(index, question, k)
            entry["rag"] = ask_rag(client, question, entry["hits"])
        if mode in (MODE_BASE, MODE_BOTH):
            entry["base"] = ask_baseline(client, question)
    st.session_state.history.append(entry)
    st.rerun()
