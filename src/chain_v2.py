import os
from dotenv import load_dotenv
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_vllm import VLLMEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from tenacity import retry, stop_after_attempt, wait_exponential
load_dotenv()

class ChainSQL:
    """SQL Query Chain that automatically initializes when instantiated"""
    
    def __init__(self, llm, db):
        """Initialize with LLM and database connection"""
        self.llm = llm.bind(
            stop=["<|im_end|>", "<|endoftext|>"],
            temperature=0.1,
            max_tokens=512
        )
        self.db = db
        self.examples = [
            {
                "input": "Có bao nhiêu loại nồi cơm điện. Trong câu có GROUP_NAME: Nồi cơm điện",
                "output": """<sql_query>
SELECT COUNT(*) 
FROM data_items 
WHERE GROUP_NAME LIKE '%Nồi cơm điện%';
</sql_query>"""
            },
            # ... (rest of examples remain the same as in chain.py)
        ]

        # Initialize the chain during construction
        self.chain = self._create_chain()

    def _create_chain(self):
        """Internal method to create the SQL query chain"""
        # Initialize vLLM embeddings
        embeddings = VLLMEmbeddings(
            model_name="vllm-embedding-model",
            api_key=os.getenv('VLLM_API_KEY')
        )

        # Create example selector
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            embeddings,
            FAISS,
            k=3,
            input_keys=["input"]
        )

        # System prompt template (same as before)
        system_prefix = """<|im_start|>system
        Bạn là chuyên gia SQL cho hệ thống bán đồ gia dụng. Hãy:
        1. Phân tích từ khóa tiếng Việt
        2. Tạo SQL query với điều kiện LIKE
        3. Đặt kết quả trong thẻ <sql_query>

        Thông tin bảng {table_info}:
        - ORDER | PRODUCT_INFO_ID | GROUP_NAME | PRODUCT_CODE
        - NAME | SPECIFICATION_BACKUP | RAW_PRICE
        - QUANTITY_SOLD | VAT_PRICE_1 | VAT_PRICE_2

        Dưới đây là các ví dụ mẫu:"""

        example_prompt = PromptTemplate.from_template(
            """<|im_start|>user
        {input}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>"""
        )

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=system_prefix + "\n\n",  # Thêm newline để tách biệt
            suffix="<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
            input_variables=["input","top_k", "table_info"]
        )

        # Tạo full prompt
        full_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Answer template
        answer_prompt = PromptTemplate.from_template(
            """<|im_start|>system
        Bạn là trợ lý chuyển đổi kết quả SQL sang câu trả lời tự nhiên. Hãy:
        1. Phân tích kết quả SQL
        2. Trình bày dạng bullet points nếu có dữ liệu
        3. Đề xuất sản phẩm thay thế khi không có kết quả
        4. Dùng tiếng Việt thân thiện<|im_end|>
        <|im_start|>user
        THÔNG TIN CẦN XỬ LÝ:
        - Câu hỏi: {question}
        - Truy vấn SQL: {query}
        - Kết quả SQL: {result}

        YÊU CẦU:
        {{
        Nếu result rỗng → 
            "Xin lỗi, hiện không có sản phẩm phù hợp. Bạn có thể tham khảo các sản phẩm tương tự: [GỢI Ý]"
        Nếu có kết quả → 
            "Tìm thấy {count} sản phẩm: [DÙNG BULLET POINTS]"
        }}<|im_end|>
        <|im_start|>assistant
        """
        )


        # Create chain components
        write_query = create_sql_query_chain(self.llm, self.db, full_prompt)
        execute_query = QuerySQLDataBaseTool(db=self.db)

        def extract(res: str) -> str:
            # Xử lý cả 2 định dạng response
            formats = [
                ("```sql", "```"),        # Format code block
                ("<sql_query>", "</sql_query>")  # Format XML
            ]
            
            for start_tag, end_tag in formats:
                if start_tag in res and end_tag in res:
                    start_idx = res.find(start_tag) + len(start_tag)
                    end_idx = res.find(end_tag, start_idx)
                    return res[start_idx:end_idx].strip()
            
            # Fallback cho raw SQL
            return res.split(";")[0] + ";" if "SELECT" in res else ""

        def clean_response(response: str) -> str:
            """Clean the final response"""
            return response.replace("**", "").strip()

        # Build and return the chain
        return (
            RunnablePassthrough.assign(query=write_query)
            .assign(result=itemgetter("query") | RunnableLambda(extract) | execute_query)
            | answer_prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(clean_response)
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def invoke(self, input_dict):
        try:
            return self.chain.invoke(input_dict)
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

