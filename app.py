import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("API_KEY")
)

SYS_PROMPT = """You are a friendly German language tutor. 
When the user sends you any message:
1. Translate it into German
2. Explain the grammar briefly in English
3. Give one example sentence in German with English translation."""

st.title("🎓 German Tutor Chatbot")
st.caption("Chat with a friendly German language tutor.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything in English..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    messages = [{"role": "system", "content": SYS_PROMPT}] + st.session_state.messages
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=500
    )
    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)