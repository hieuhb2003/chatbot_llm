import gradio as gr
from src.service_v2 import ChatService
import os
from dotenv import load_dotenv
import uuid
from langchain_openai import ChatOpenAI
load_dotenv()
os.environ['VLLM_API_KEY'] = os.getenv('VLLM_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'


def create_chat_service():
    llm = ChatOpenAI(
        model="Qwen/Qwen1.5-7B-Chat",
        openai_api_base="http://localhost:8000/v1",  # URL vLLM server
        api_key="EMPTY",
        max_tokens=1024
    )
    return ChatService(llm)  # Giả sử ChatService đã được sửa để nhận llm

chat_service = create_chat_service()

chat_history = []

def chat(question, session_id):
    try:
        response = chat_service.process_question(question, session_id)
        return response
    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}"

def start_chatbot():
    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot Hỏi Đáp (vLLM Version)")
        
        session_id = gr.Textbox(visible=False, value=str(uuid.uuid4()))
        
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
            inputs=[question_input, session_id],
            outputs=[chat_history_output]
        )

    return demo

if __name__ == "__main__":
    app = start_chatbot()
    app.launch(share=True)
