import streamlit as st
import ollama
import os

st.set_page_config(page_title="Thai Festivals AI", page_icon="🪔")

st.title("🪔 Thai Festivals AI Assistant (Local LLM)")
st.caption("Does giving a local LLM local documents make it more accurate? Songkran, Loy Krathong, and more.")

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
                # Call local LLM (e.g., Llama 3.2 or Mistral via Ollama)
                response = ollama.generate(model="llama3.2", prompt=grounded_prompt)
                answer = response["response"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error calling Local LLM: {e}")
