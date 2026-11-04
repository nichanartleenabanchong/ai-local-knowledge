import streamlit as st
import google.generativeai as genai
import os

st.set_page_config(page_title="Thai Festivals AI", page_icon="🪔")

st.title("🪔 Thai Festivals AI Assistant")
st.caption("Does giving an AI local documents make it more accurate? Songkran, Loy Krathong and more.")

# ดึง Gemini API Key จาก Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets. Please check your settings.")
    st.stop()

genai.configure(api_key=api_key)

# อ่านไฟล์ข้อมูล local จากโฟลเดอร์ docs หรือไฟล์ chula_knowledge.txt
local_knowledge = ""
if os.path.exists("docs/loy_krathong.md"):
    with open("docs/loy_krathong.md", "r", encoding="utf-8") as f:
        local_knowledge += f.read() + "\n\n"

if os.path.exists("chula_knowledge.txt"):
    with open("chula_knowledge.txt", "r", encoding="utf-8") as f:
        local_knowledge += f.read()

# เปลี่ยนมาใช้รุ่น gemini-2.5-flash
model = genai.GenerativeModel("gemini-2.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Songkran, Loy Krathong, dates, rituals, regional differences..."):
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
        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(grounded_prompt)
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
