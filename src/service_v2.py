from langchain_openai import ChatOpenAI
from config import Config
from src.ner import NER
from src.chain import ChainSQL
from rag.document_retrieval import DocumentRetrieval
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, llm):
        self.llm = llm.bind(
            stop=["<|im_end|>", "<|endoftext|>"],
            temperature=0.1,
            max_tokens=1024
        )
        self.db = SQLDatabase.from_uri(Config.db_sql)
        self.model = NER()
        self.chain = ChainSQL(self.llm, self.db)
        self.retrieval_doc = DocumentRetrieval()
        self.sessions = {}
        self.session_timeout = timedelta(minutes=30)

    def get_session(self, session_id):
        """Quản lý phiên làm việc và tự động dọn dẹp"""
        self._cleanup_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "chat_history": [],
                "created_at": datetime.now()
            }
        return self.sessions[session_id]

    def _cleanup_sessions(self):
        """Xóa các phiên không hoạt động"""
        now = datetime.now()
        expired = [sid for sid, s in self.sessions.items() 
                  if (now - s['created_at']) > self.session_timeout]
        for sid in expired:
            del self.sessions[sid]

    def rewrite_question(self, question, session_id):
        """Cải thiện câu hỏi với định dạng ChatML"""
        session = self.get_session(session_id)
        prompt = ChatPromptTemplate.from_template("""
            <|im_start|>system
            Bạn là trợ lý cải thiện câu hỏi. Hãy:
            - Sửa lỗi chính tả
            - Làm rõ ngữ nghĩa
            - Bổ sung ngữ cảnh từ lịch sử<|im_end|>
            <|im_start|>user
            LỊCH SỬ: {history}
            CÂU HỎI GỐC: {question}
            Câu hỏi sau khi cải thiện:<|im_end|>
            <|im_start|>assistant
        """)
        
        chain = prompt | self.llm
        return chain.invoke({
            "history": "\n".join([f"{msg['HumanMessage']} -> {msg['AIMessage']}" 
                                for msg in session['chat_history'][-3:]]),
            "question": question
        }).content

    def is_sales_related(self, question):
        """Phân loại câu hỏi với định dạng ChatML"""
        prompt = ChatPromptTemplate.from_template("""
            <|im_start|>system
            Phân loại câu hỏi có liên quan đến sản phẩm/bán hàng không.
            Chỉ trả lời "CÓ" hoặc "KHÔNG".<|im_end|>
            <|im_start|>user
            {question}<|im_end|>
            <|im_start|>assistant
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"question": question}).content
        return "CÓ" in response

    def process_question(self, question, session_id):
        """Xử lý câu hỏi chính với quản lý phiên"""
        try:
            session = self.get_session(session_id)
            
            if not self.is_sales_related(question):
                response = self.llm.invoke(question).content
                self._update_chat_history(question, response, session_id)
                return f"{response}\n\nXin mời bạn đưa ra câu hỏi về sản phẩm."

            name_entity = self.model.predict(question)
            history = self._prepare_chat_history(session_id)
            
            if 0 < len(name_entity['NAME']) <= 2:
                return self._handle_name_entities(question, name_entity, history, session_id)
            elif len(name_entity['GROUP_NAME']) > 0:
                return self._handle_group_entities(question, name_entity, history, session_id)
            else:
                return self._handle_general_case(question, history, session_id)
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return f"Xin lỗi, có lỗi xảy ra: {str(e)}"

    def _prepare_chat_history(self, session_id):
        session = self.get_session(session_id)
        return [
            msg for entry in session['chat_history'][-5:]
            for msg in (entry["HumanMessage"], entry["AIMessage"])
        ]

    def _handle_name_entities(self, question, name_entity, history, session_id):
        context = '\n'.join(self.retrieval_doc.retrieve_documents(item) for item in name_entity['NAME'])
        response = self.chain.invoke({
            "question": question,
            "input": f"{question}\nContext: {context}",
            "top_k": 3,
            "table_info": "data_items",
            "history": history
        }).content
        self._update_chat_history(question, response, session_id)
        return response

    def _handle_group_entities(self, question, name_entity, history, session_id):
        new_question = f"{question} [GROUP_NAME: {', '.join(name_entity['GROUP_NAME'])}]"
        if name_entity['NAME']:
            new_question += f" [NAME: {', '.join(name_entity['NAME'])}]"
        
        response = self.chain.invoke({
            "question": new_question,
            "input": new_question,
            "top_k": 3,
            "table_info": "data_items",
            "history": history
        }).content
        self._update_chat_history(question, response, session_id)
        return response

    def _handle_general_case(self, question, history, session_id):
        response = self.chain.invoke({
            "question": question,
            "input": question,
            "top_k": 3,
            "table_info": "data_items",
            "history": history
        }).content
        self._update_chat_history(question, response, session_id)
        return response

    def _update_chat_history(self, question, response, session_id):
        session = self.get_session(session_id)
        session['chat_history'].append({
            "HumanMessage": question,
            "AIMessage": response
        })
        if len(session['chat_history']) > 20:
            session['chat_history'].pop(0)