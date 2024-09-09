import gradio as gr
import uuid
from config import Config
from ner.ner import NER
from sql_qa.chain import ChainSQL
from rag.document_retrieval import DocumentRetrieval
import os
from history.history import HistoryProcessor
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2']  = 'true'
from config import Config
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from config import Config
import os
from app.main import answer

llm = ChatGroq(model="gemma2-9b-it")
# llm = ChatGroq(model="llama-3.1-70b-versatile")
db = SQLDatabase.from_uri(Config.db_sql)
model = NER()
chain = ChainSQL().create_chain(llm, db)
retrieval_doc = DocumentRetrieval()

def create_new_conversation():
    return str(uuid.uuid4())[:8]  

def chat(question, id_user, id_conversation):
    if not id_conversation:  # Create new conversation ID if none exists
        id_conversation = create_new_conversation()

    response = answer(question, id_user, id_conversation, llm, model, chain, retrieval_doc)

    history_processor = HistoryProcessor()
    chat_history_data = history_processor.load_history(id_user, id_conversation)
    
    formatted_history = []
    for message in chat_history_data:
        formatted_history.append((message["HumanMessage"], message["AIMessage"]))

    return id_conversation, formatted_history

def start_chatbot():
    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot Hỏi Đáp")

        id_user_input = gr.Textbox(label="Nhập ID của bạn", placeholder="ID người dùng")
        id_conversation_input = gr.Textbox(label="Nhập hoặc tạo mới ID Hội Thoại", placeholder="ID hội thoại")

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
            inputs=[question_input, id_user_input, id_conversation_input],
            outputs=[id_conversation_input, chat_history_output]
        )

    return demo

if __name__ == "__main__":
    app = start_chatbot()
    app.launch()