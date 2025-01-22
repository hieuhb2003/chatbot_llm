import uuid
from config import Config
from src.ner import NER
from src.chain import ChainSQL
from rag.document_retrieval import DocumentRetrieval
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class ChatService:
    def __init__(self):
        self.llm = ChatGroq(model="gemma2-9b-it")
        self.db = SQLDatabase.from_uri(Config.db_sql)
        self.model = NER()
        self.chain = ChainSQL(self.llm, self.db)
        self.retrieval_doc = DocumentRetrieval()
        self.session_id = str(uuid.uuid4())[:8]
        self.chat_history = []

    def rewrite_question(self, question):
        """Rewrite user's question to be more clear and complete using chat history"""
        prompt = ChatPromptTemplate.from_template("""
            Bạn là trợ lý viết lại câu hỏi. Hãy cải thiện câu hỏi sau dựa trên ngữ cảnh hội thoại:
            - Sửa lỗi chính tả nếu có
            - Làm rõ nghĩa câu hỏi
            - Giữ nguyên ý định ban đầu
            - Thêm thông tin từ lịch sử hội thoại nếu cần
            
            Lịch sử hội thoại gần đây:
            {history}
            
            Câu hỏi cần cải thiện:
            {question}
            
            Câu hỏi sau khi cải thiện:
        """)
        
        chain = prompt | self.llm
        return chain.invoke({
            "history": "\n".join([f"{msg['HumanMessage']} -> {msg['AIMessage']}" 
                                 for msg in self.chat_history[-3:]]),
            "question": question
        })

    def is_sales_related(self, question):
        """Determine if the question is related to sales using LLM"""
        prompt = ChatPromptTemplate.from_template("""
            Bạn là trợ lý phân loại câu hỏi. Hãy xác định xem câu hỏi sau có liên quan đến bán hàng, các câu hỏi liên quan đến sản phẩm đồ gia dụng không.
            Trả lời chỉ bằng "CÓ" hoặc "KHÔNG".
            
            Câu hỏi: {question}
            
            Trả lời:
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"question": question})
        return "CÓ" in response.content

    def process_question(self, question):
        # First check if this is a sales-related question
        if not self.is_sales_related(question):
            # If not sales-related, answer the question and suggest product-related questions
            response = self.llm.invoke(question)
            return f"{response.content}\n\nXin mời bạn đưa ra câu hỏi về sản phẩm cho tôi."

        # Process question and get response
        name_entity = self.model.predict(question)
        
        # Get last 5 messages from history
        format_chat_his = []
        for i in range(max(0, len(self.chat_history) - 5), len(self.chat_history)):
            format_chat_his.append(self.chat_history[i]["HumanMessage"])
            format_chat_his.append(self.chat_history[i]["AIMessage"])

        response = ""
        if len(name_entity['NAME']) <= 2 and len(name_entity['NAME']) > 0:
            context = ""
            for item in name_entity['NAME']:
                context += self.retrieval_doc.retrieve_documents(item) + '\n'
            
            response = self.chain.invoke({
                "question": question,
                "input": question,
                "top_k": 3,
                "table_info": "data_items",
                "history": format_chat_his
            })
            
        elif len(name_entity['GROUP_NAME']) > 0:
            new_question = question + " .Trong câu có GROUP_NAME: "
            for item in name_entity['GROUP_NAME']:
                new_question += item + " "
            if len(name_entity['NAME']) > 0:
                new_question += " .Trong câu có NAME: "
                for item in name_entity['NAME']:
                    new_question += item + " "
            
            response = self.chain.invoke({
                "question": new_question,
                "input": new_question,
                "top_k": 3,
                "table_info": "data_items",
                "history": format_chat_his
            })
        else:
            response = self.chain.invoke({
                "question": question,
                "input": question,
                "top_k": 3,
                "table_info": "data_items",
                "history": format_chat_his
            })

        # Update in-memory history
        self.chat_history.append({
            "HumanMessage": question,
            "AIMessage": response
        })
        
        return response
