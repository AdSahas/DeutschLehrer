import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
import os

client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.3",
    token=os.getenv("API_KEY")
)

SYS_PROMPT = """

You are a friendly German language tutor. 
When the user sends you any message:
1. Translate it into German
2. Explain the grammar briefly in English
3. Give one example sentence in German with English translation.

"""

def chat(message, history):
    messages = [{"role": "system", "content": SYS_PROMPT}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(messages=messages, max_tokens=500)
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn = chat,
    title = "German Tutor Chatbot",
    description = "Chat with a friendly German language tutor. Ask any question or send any message, and the tutor will translate it into German, explain the grammar, and provide an example sentence.",

)

demo.launch(server_name="0.0.0.0")