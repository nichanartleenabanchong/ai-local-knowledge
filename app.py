import streamlit as st
import os
from groq import Groq

st.set_page_config(page_title="Thai Festivals AI", page_icon="🪔")

st.title("🪔 Thai Festivals AI Assistant (Open-Source LLM)")
st.caption("Local RAG implementation using Meta's open-source Llama 3.3 model.")

# Fetch Groq API Key
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets. Please check your app settings.")
    st.stop()

client = Groq(api_key=api_key)

# Load local knowledge documents
local_knowledge = ""
if os.path.exists("docs/loy_krathong.md"):
    with open("docs/loy_krathong.md", "r", encoding="utf-8") as f:
        local_knowledge += f.read() + "\n\n"

if os.path.exists("chula_knowledge.txt"):
    with open("chula_knowledge.txt", "r", encoding="utf-8") as f:
        local_knowledge += f.read()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Songkran, Loy Krathong, dates, rituals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    grounded_prompt = f"""
    You are an expert AI assistant specializing in Thai festivals and traditions.
    Answer the user's question accurately based ONLY on the provided reference text below. 
    If the answer cannot be found in the text, politely state that you don't know based on the local documents.

    --- Reference Documents ---
    {local_knowledge}
    ---------------------------

    User Question: {prompt}
    """

    with st.chat_message("assistant"):
        with st.spinner("Llama 3.3 thinking..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": grounded_prompt}],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
