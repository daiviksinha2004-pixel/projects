import gradio as gr
from transformers import pipeline
from gtts import gTTS

text_generator = pipeline("text2text-generation", model="google/flan-t5-small")

def chat_with_jarvis(prompt):
    response = text_generator(prompt, max_length=100)[0]["generated_text"]
    tts = gTTS(text=response)
    tts.save("response.mp3")
    return response, "response.mp3"

interface = gr.Interface(
    fn=chat_with_jarvis,
    inputs=gr.Textbox(lines=2, placeholder="Ask Jarvis..."),
    outputs=[gr.Textbox(label="Response"), gr.Audio(label="Voice")],
    title="Jarvis AI Assistant",
    description="Ask anything. Get a voice reply powered by FLAN-T5 + gTTS."
)

if __name__ == "__main__":
    interface.launch()
