import os
from dotenv import load_dotenv
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
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

load_dotenv()

class SQLQueryChain:
    """A class for generating and executing SQL queries based on natural language input."""
    
    def __init__(self):
        # Example queries for few-shot learning
        self.examples = [
            {
                "input": "Có bao nhiêu loại nồi cơm điện. Trong câu có GROUP_NAME: Nồi cơm điện",
                "output": """<sql_query>
SELECT COUNT(*) 
FROM data_items 
WHERE GROUP_NAME LIKE '%Nồi cơm điện%';
</sql_query>"""
            },
            {
                "input": "So sánh Ghế Massage Makano MKGM-10003 với Ghế Massage Daikiosan DKGM-20006. Trong câu có NAME: Ghế Massage Daikiosan DKGM-20006, Ghế Massage Makano MKGM-10003",
                "output": """<sql_query>
SELECT * 
FROM data_items 
WHERE NAME LIKE '%Ghế Massage Makano MKGM-10003%' 
    OR NAME LIKE '%Ghế Massage Daikiosan DKGM-20006%' 
    OR NAME LIKE '%MKGM-10003%' 
    OR NAME LIKE '%DKGM-20006%';
</sql_query>"""
            },
            {
                "input": "Máy giặt nào rẻ nhất. Trong câu có GROUP_NAME: Máy Giặt",
                "output": """<sql_query>
SELECT NAME, PRICE 
FROM data_items 
WHERE GROUP_NAME LIKE '%Máy giặt%' 
    OR SPECIFICATION_BACKUP LIKE '%Máy giặt%' 
    OR NAME LIKE '%máy giặt%' 
ORDER BY RAW_PRICE ASC 
LIMIT 1;
</sql_query>"""
            },
            {
                "input": "Công suất của Bàn Ủi Khô Bluestone DIB-3726 1300W. Trong câu có NAME: Bàn Ủi Khô Bluestone DIB-3726 1300W",
                "output": """<sql_query>
SELECT NAME, SPECIFICATION_BACKUP 
FROM data_items 
WHERE NAME LIKE '%Bàn Ủi Khô Bluestone DIB-3726 1300W%' 
    OR NAME LIKE '%DIB-3726%' 
    OR SPECIFICATION_BACKUP LIKE '%Bàn Ủi Khô Bluestone DIB-3726 1300W%' 
    OR SPECIFICATION_BACKUP LIKE '%DIB-3726%' 
LIMIT 1;
</sql_query>"""
            },
            {
                "input": "Lò Vi Sóng Bluestone MOB-7716 có thể hẹn giờ trong bao lâu. Trong câu có NAME: Lò Vi Sóng Bluestone MOB-7716 có nướng 20 lít",
                "output": """<sql_query>
SELECT NAME, SPECIFICATION_BACKUP 
FROM data_items 
WHERE NAME LIKE '%Lò Vi Sóng Bluestone MOB-7716 có nướng 20 lít%' 
    OR NAME LIKE '%MOB-7716%' 
    OR SPECIFICATION_BACKUP LIKE '%Lò Vi Sóng Bluestone MOB-7716 có nướng 20 lít%' 
    OR SPECIFICATION_BACKUP LIKE '%MOB-7716%' 
LIMIT 1;
</sql_query>"""
            },
            {
                "input": "Máy NLMT Empire 180 Lít Titan M&EGD000224 có bao nhiêu ống. Trong câu có NAME: Máy NLMT Empire 180 Lít Titan M&EGD000224",
                "output": """<sql_query>
SELECT SPECIFICATION_BACKUP 
FROM data_items 
WHERE NAME LIKE '%M&EGD000224%' 
    OR NAME LIKE '%Máy NLMT Empire 180 Lít Titan M&EGD000224%' 
    OR SPECIFICATION_BACKUP LIKE '%Máy NLMT Empire 180 Lít Titan M&EGD000224%'  
    OR SPECIFICATION_BACKUP LIKE '%M&EGD000224%' 
LIMIT 1;
</sql_query>"""
            },
            {
                "input": "có hình ảnh nồi KL-619 không. Trong câu có NAME: Nồi cơm điện KALite KL-619, dung tích 1,8 lít",
                "output": """<sql_query>
SELECT * 
FROM data_items 
WHERE NAME LIKE '%KL-619%' 
    OR NAME LIKE '%Nồi cơm điện KALite KL-619, dung tích 1,8 lít%' 
LIMIT 1;
</sql_query>"""
            },
            {
                "input": "Đèn đường năng lượng mặt trời SUNTEK S500 PLUS, công suất 500W đắt quá, có cái nào rẻ hơn không. Trong câu có NAME: Đèn đường năng lượng mặt trời SUNTEK S500 PLUS, công suất 500W",
                "output": """<sql_query>
SELECT * 
FROM (
    SELECT * 
    FROM data_items 
    WHERE NAME LIKE '%Đèn đường năng lượng mặt trời SUNTEK S500 PLUS%' 
    UNION 
    SELECT * 
    FROM (
        SELECT * 
        FROM data_items 
        WHERE NAME LIKE '%Đèn đường năng lượng mặt trời%' 
        ORDER BY RAW_PRICE ASC 
        LIMIT 3
    )
) AS combined_results;
</sql_query>"""
            },
            {
                "input": "Máy giặt lồng dọc có thông số như thế nào. Trong câu có GROUP_NAME: Máy Giặt",
                "output": """<sql_query>
SELECT NAME, SPECIFICATION_BACKUP 
FROM data_items 
WHERE NAME LIKE '%máy giặt lồng dọc%' 
    OR GROUP_NAME LIKE '%Máy Giặt%' 
    OR SPECIFICATION_BACKUP LIKE '%máy giặt lồng dọc%' 
    OR NAME LIKE '%Máy giặt lồng dọc%' 
    OR SPECIFICATION_BACKUP LIKE '%Máy giặt lồng dọc%' 
LIMIT 3;
</sql_query>"""
            },
            {
                "input": "Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE đã bán bao nhiêu sản phẩm. Trong câu có NAME: Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE",
                "output": """<sql_query>
SELECT QUANTITY_SOLD 
FROM data_items 
WHERE NAME LIKE '%Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE%' 
    OR SPECIFICATION_BACKUP LIKE '%Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE%' 
LIMIT 1;
</sql_query>"""
            },
            {
                "input": "Giá gốc của sản phẩm Bếp từ đơn AIO Smart kèm nồi. Trong câu có NAME: Bếp từ đơn AIO Smart kèm nồi",
                "output": """<sql_query>
SELECT RAW_PRICE 
FROM data_items 
WHERE NAME LIKE '%Bếp từ đơn AIO Smart kèm nồi%' 
    OR SPECIFICATION_BACKUP LIKE '%Bếp từ đơn AIO Smart kèm nồi%' 
LIMIT 1;
</sql_query>"""
            },
            {
                "input": "Tôi muốn mua điều hòa dưới 10 triệu và Đèn Năng Lượng Mặt Trời trên 8 triệu. Trong câu có: GROUP_NAME: Đèn Năng Lượng Mặt Trời, Điều hòa",
                "output": """<sql_query>
SELECT * 
FROM data_items 
WHERE (GROUP_NAME LIKE '%điều hòa%' AND RAW_PRICE < 10000000) 
    OR (GROUP_NAME LIKE '%Đèn Năng Lượng Mặt Trời%' AND RAW_PRICE > 8000000);
</sql_query>"""
            },
            {
                "input": "Bán cho tôi 2 điều hòa Daikin có giá rẻ, 3 nồi cơm điện giá có tầm giá chung",
                "output": """<sql_query>
SELECT * 
FROM data_items 
WHERE (NAME LIKE '%điều hòa Daikin%' AND RAW_PRICE IS NOT NULL) 
    OR (NAME LIKE '%nồi cơm điện%' AND RAW_PRICE IS NOT NULL) 
ORDER BY 
    CASE 
        WHEN NAME LIKE '%điều hòa Daikin%' THEN RAW_PRICE 
        WHEN NAME LIKE '%nồi cơm điện%' THEN RAW_PRICE 
    END ASC 
LIMIT 5;
</sql_query>"""
            },
            {
                "input": "Bếp từ nào là sản phẩm bán chạy nhất. Trong câu có GROUP_NAME: Bếp từ",
                "output": """<sql_query>
SELECT NAME, QUANTITY_SOLD 
FROM data_items 
WHERE GROUP_NAME LIKE '%Bếp từ%' 
    OR NAME LIKE '%bếp từ%' 
    OR SPECIFICATION_BACKUP LIKE '%Bếp từ%' 
ORDER BY QUANTITY_SOLD DESC 
LIMIT 1;
</sql_query>"""
            }
        ]


    def create_query_chain(self, llm, db):
        """Create and configure the SQL query generation chain.
        
        Args:
            llm: Language model instance
            db: SQL database connection
            
        Returns:
            Configured SQL query chain
        """
        # Initialize text embeddings for example selection
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Configure example selector for few-shot learning
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            embeddings,
            FAISS,
            k=3,
            input_keys=["input"]
        )

        # System prompt template
        system_prompt = """Bạn là một chuyên gia SQLite. Luôn nhớ các thông tin bạn có thể cung cấp được liên quan đến thiết bị điện, điện tử, đồ gia dụng... hoặc sản phẩm tương tự. Từ một câu hỏi đầu vào, hãy tạo một truy vấn SQLite đúng về mặt cú pháp để chạy, nếu có lịch sử cuộc trò truyền thì hãy dựa vào đó để tạo truy vấn SQLite đúng với ngữ cảnh khi đó.

Đây là thông tin về bảng {table_info} bao gồm các cột: ORDER, PRODUCT_INFO_ID, GROUP_NAME, PRODUCT_CODE, NAME, SPECIFICATION_BACKUP, NON_VAT_PRICE_1, VAT_PRICE_1, COMMISSION_1, THRESHOLD_1, NON_VAT_PRICE_2, VAT_PRICE_2, COMMISSION_2, THRESHOLD_2, NON_VAT_PRICE_3, VAT_PRICE_3, COMMISSION_3, RAW_PRICE, QUANTITY_SOLD.

Trả về tối đa {top_k} kết quả trừ khi có yêu cầu khác. Lưu ý rằng, câu SQL query sẽ được để trong thẻ <sql_query>...</sql_query> để dễ dàng trích xuất.

Dưới đây là một số ví dụ tương tự với câu hỏi của bạn:"""

        # Configure prompt templates
        example_prompt = PromptTemplate.from_template(
            "User input: {input}\nSQL query: {output}"
        )

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=system_prompt,
            suffix="User input: {input}\nSQL query: ",
            input_variables=["input", "top_k", "table_info"]
        )

        # Create full prompt chain
        full_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Configure answer generation template
        answer_prompt = PromptTemplate.from_template(
            """Với câu hỏi của người dùng sau đây, truy vấn SQL tương ứng, và kết quả SQL, hãy trả lời câu hỏi của người dùng.
            Câu hỏi: {question}
            Truy vấn SQL: {query}
            Kết quả SQL: {result}
            Nếu kết quả SQL trả về rỗng, hãy thông báo cho người dùng rằng không có sản phẩm phù hợp với yêu cầu của họ và yêu cầu họ cung cấp thông tin cụ thể hơn hoặc gợi ý tư vấn giải pháp giúp họ.
            Câu trả lời: """
        )

        # Create chain components
        write_query = create_sql_query_chain(llm, db, full_prompt)
        execute_query = QuerySQLDataBaseTool(db=db)

        def extract_query(response: str) -> str:
            """Extract SQL query from response text.
            
            Args:
                response: Text containing SQL query
                
            Returns:
                Extracted SQL query string
            """
            # Extract from XML tags if present
            start_tag = "<sql_query>"
            end_tag = "</sql_query>"
            
            if start_tag in response and end_tag in response:
                start_index = response.find(start_tag) + len(start_tag)
                end_index = response.find(end_tag)
                return response[start_index:end_index].strip()
            
            # Fallback to direct SQL extraction
            start_index = response.find("SELECT")
            if start_index == -1:
                return ""
            
            query = ""
            i = start_index
            while i < len(response) and response[i] != ";":
                query += response[i]
                i += 1
            return query + ";"

        def clean_response(response: str) -> str:
            """Clean and format final response.
            
            Args:
                response: Raw response text
                
            Returns:
                Cleaned response string
            """
            return response.replace("**", "").strip()

        # Build and return the complete chain
        return (
            RunnablePassthrough.assign(query=write_query)
            .assign(result=itemgetter("query") | RunnableLambda(extract_query) | execute_query)
            | answer_prompt
            | llm
            | StrOutputParser()
            | RunnableLambda(clean_response)
        )
