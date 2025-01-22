import gradio as gr
from src.service import ChatService
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

chat_service = ChatService()

chat_history = []

def chat(question):
    response = chat_service.process_question(question)
    chat_history.append((question, response))
    return chat_history

def start_chatbot():
    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot Hỏi Đáp")

        with gr.Row():
            with gr.Column(scale=4):
                chat_history_output = gr.Chatbot(label="Lịch sử hội thoại")

        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(label="Câu hỏi của bạn", placeholder="Nhập câu hỏi...")

        with gr.Row():
            with gr.Column(scale=4):
                submit_button = gr.Button("Gửi câu hỏi")

        submit_button.click(
            fn=chat,
            inputs=[question_input],
            outputs=[chat_history_output]
        )

    return demo

if __name__ == "__main__":
    app = start_chatbot()
    app.launch(share=True)
