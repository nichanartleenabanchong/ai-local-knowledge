"""
Thai Festivals Chatbot — Streamlit + local LLM (Ollama)
Run:  streamlit run app.py
"""
import requests
import streamlit as st

from rag import DEFAULT_MODEL, Retriever, answer, load_chunks

st.set_page_config(page_title="Thai Festivals Chatbot", page_icon="🏮")
st.title("🏮 Thai Festivals & Traditions Chatbot")
st.caption("Local LLM (Ollama) · compare answers WITH vs WITHOUT our festival documents")


@st.cache_resource
def get_retriever():
    return Retriever(load_chunks())


retriever = get_retriever()

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", DEFAULT_MODEL)
    mode = st.radio("Mode", ["With documents (RAG)", "Without documents", "Compare both"])
    k = st.slider("Number of passages to use", 1, 8, 4)
    st.markdown(f"📄 Loaded **{len(retriever.chunks)}** passages from `docs/`")
    if st.button("Clear chat"):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []


def show_sources(hits):
    with st.expander("Sources used"):
        for h in hits:
            st.markdown(f"**{h['source']}** (score {h['score']:.2f})")
            st.text(h["text"])


def render(entry):
    if entry["mode"] == "Compare both":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📚 With documents**")
            st.markdown(entry["rag"])
            show_sources(entry["hits"])
        with c2:
            st.markdown("**🧠 Without documents**")
            st.markdown(entry["plain"])
    else:
        st.markdown(entry["answer"])
        if entry.get("hits"):
            show_sources(entry["hits"])


# ---------- Show chat history ----------
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        render(entry)

# ---------- New question ----------
question = st.chat_input("Ask about Songkran, Loy Krathong, Yi Peng...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                entry = {"question": question, "mode": mode}
                if mode == "Compare both":
                    entry["rag"], entry["hits"] = answer(question, True, retriever, model, k)
                    entry["plain"], _ = answer(question, False, retriever, model, k)
                else:
                    use_docs = mode.startswith("With documents")
                    entry["answer"], entry["hits"] = answer(question, use_docs, retriever, model, k)
            render(entry)
            st.session_state.history.append(entry)
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Ollama. Start it first (open the Ollama app or run `ollama serve`).")
        except requests.exceptions.HTTPError as e:
            st.error(f"Ollama error: {e}. Did you run `ollama pull {model}`?")
