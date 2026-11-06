import streamlit as st
import os
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Thai Festivals AI", page_icon="🪔")

st.title("🪔 Thai Festivals AI Assistant (Local/Open-Source LLM)")
st.caption("Local RAG implementation using an open-source LLM (Qwen2.5-7B-Instruct).")

# Initialize Hugging Face open-source inference client
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

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
        with st.spinner("Local LLM thinking..."):
            try:
                # Query open-source local LLM
                response = client.text_generation(
                    grounded_prompt,
                    max_new_tokens=512,
                    temperature=0.3
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
