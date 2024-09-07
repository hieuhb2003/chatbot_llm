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
from langchain_core.prompts import (
    ChatPromptTemplate
)
from langchain_community.utilities.sql_database import SQLDatabase
from config import Config
import os
from app.main import answer

llm = ChatGroq(model="gemma2-9b-it")
db = SQLDatabase.from_uri(Config.db_sql)
model = NER()
chain = ChainSQL().create_chain(llm, db)
retrieval_doc = DocumentRetrieval()

import gradio as gr
import uuid
from config import Config
from ner.ner import NER
from sql_qa.chain import ChainSQL
from rag.document_retrieval import DocumentRetrieval
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2']  = 'true'
from config import Config
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate
)
from langchain_community.utilities.sql_database import SQLDatabase
from config import Config
import os
from app.main import answer

llm = ChatGroq(model="gemma2-9b-it")
db = SQLDatabase.from_uri(Config.db_sql)
model = NER()
chain = ChainSQL().create_chain(llm, db)
retrieval_doc = DocumentRetrieval()

# # Function to create a new conversation ID
# def create_new_conversation():
#     return str(uuid.uuid4())[:8]  # Simpler 8-character ID

# # Function to handle the question and response
# def chat(question, id_user, id_conversation, chat_history):
#     if not id_conversation:  # Create new conversation ID if none exists
#         id_conversation = create_new_conversation()

#     # Process and generate the response
#     response = answer(question, id_user, id_conversation, llm, model, chain, retrieval_doc)

#     # Append the latest question and response to chat history
#     chat_history.append((question, response))
#     return id_conversation, chat_history

# # Gradio Interface
# def start_chatbot():
#     with gr.Blocks() as demo:
#         gr.Markdown("# Chatbot Hỏi Đáp")

#         # Input for user ID
#         id_user_input = gr.Textbox(label="Nhập ID của bạn", placeholder="ID người dùng")

#         # Editable input for conversation ID (user can input an existing ID)
#         id_conversation_input = gr.Textbox(label="Nhập hoặc tạo mới ID Hội Thoại", placeholder="ID hội thoại")

#         # Chatbot to display conversation history and accept new question
#         with gr.Row():
#             with gr.Column(scale=4):
#                 chat_history_output = gr.Chatbot(label="Lịch sử hội thoại")

#             with gr.Column(scale=1):
#                 question_input = gr.Textbox(label="Câu hỏi của bạn", placeholder="Nhập câu hỏi...")

#         # Button to submit the question
#         submit_button = gr.Button("Gửi câu hỏi")

#         # When "Gửi câu hỏi" is clicked
#         submit_button.click(
#             fn=chat,
#             inputs=[question_input, id_user_input, id_conversation_input, chat_history_output],
#             outputs=[id_conversation_input, chat_history_output]
#         )

#     return demo

# # Launch Gradio UI
# if __name__ == "__main__":
#     app = start_chatbot()
#     app.launch()

def create_new_conversation():
    return str(uuid.uuid4())[:8]  # Simpler 8-character ID

# Function to handle the question and response
def chat(question, id_user, id_conversation):
    if not id_conversation:  # Create new conversation ID if none exists
        id_conversation = create_new_conversation()

    # Process and generate the response
    response = answer(question, id_user, id_conversation, llm, model, chain, retrieval_doc)

    # Load previous conversation history
    history_processor = HistoryProcessor()
    chat_history_data = history_processor.load_history(id_user, id_conversation)
    
    # Format the loaded chat history into the Chatbot format
    formatted_history = []
    for message in chat_history_data:
        formatted_history.append((message["HumanMessage"], message["AIMessage"]))

    # Append the latest question and response to chat history
    formatted_history.append((question, response))
    
    return id_conversation, formatted_history

# Gradio Interface
def start_chatbot():
    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot Hỏi Đáp")

        # Input for user ID
        id_user_input = gr.Textbox(label="Nhập ID của bạn", placeholder="ID người dùng")

        # Editable input for conversation ID (user can input an existing ID)
        id_conversation_input = gr.Textbox(label="Nhập hoặc tạo mới ID Hội Thoại", placeholder="ID hội thoại")

        # Chatbot to display conversation history and accept new question
        with gr.Row():
            with gr.Column(scale=4):
                chat_history_output = gr.Chatbot(label="Lịch sử hội thoại")

        # Question input box and submit button placed under the history
        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(label="Câu hỏi của bạn", placeholder="Nhập câu hỏi...")

        # Button to submit the question
        with gr.Row():
            with gr.Column(scale=4):
                submit_button = gr.Button("Gửi câu hỏi")

        # Button to submit the question
        # submit_button = gr.Button("Gửi câu hỏi")

        # When "Gửi câu hỏi" is clicked, it calls the `chat` function
        submit_button.click(
            fn=chat,
            inputs=[question_input, id_user_input, id_conversation_input],
            outputs=[id_conversation_input, chat_history_output]
        )

    return demo

# Launch Gradio UI
if __name__ == "__main__":
    app = start_chatbot()
    app.launch()