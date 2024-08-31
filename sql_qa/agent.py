import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
from config import Config
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
import os
from langchain_community.vectorstores.faiss import FAISS

llm = ChatGroq(model="llama3-8b-8192")
db = SQLDatabase.from_uri(Config.db_sql)
examples = [
            {
                "input": "Có bao nhiêu loại nồi cơm điện. Trong câu có GROUP_NAME: Nồi cơm điện",
                "query": """SELECT COUNT(*) \nFROM data_items \nWHERE GROUP_NAME LIKE \'%Nồi cơm điện%\';"""},
            {
                "input": "So sánh Ghế Massage Makano MKGM-10003 với Ghế Massage Daikiosan DKGM-20006. Trong câu có NAME: Ghế Massage Daikiosan DKGM-20006, Ghế Massage Makano MKGM-10003",
                "query": "SELECT * \nFROM data_items \nWHERE NAME LIKE '%Ghế Massage Makano MKGM-10003%' OR NAME LIKE '%Ghế Massage Daikiosan DKGM-20006%' OR NAME LIKE '%MKGM-10003%' OR NAME LIKE '%DKGM-20006%';",
            },
            {
                "input": "Máy giặt nào rẻ nhất. Trong câu có GROUP_NAME: Máy Giặt",
                "query": "SELECT NAME, PRICE\nFROM data_items\nWHERE GROUP_NAME LIKE '%Máy giặt%' OR SPECIFICATION_BACKUP LIKE '%Máy giặt%' OR NAME LIKE \'%máy giặt%\' \nORDER BY RAW_PRICE ASC\nLIMIT 1;",
            },
            {
                "input": "Công suất của Bàn Ủi Khô Bluestone DIB-3726 1300W. Trong câu có NAME: Bàn Ủi Khô Bluestone DIB-3726 1300W",
                "query": "SELECT NAME, SPECIFICATION_BACKUP \nFROM data_items \nWHERE NAME LIKE '%Bàn Ủi Khô Bluestone DIB-3726 1300W%' OR NAME LIKE \'%DIB-3726%\' OR SPECIFICATION_BACKUP LIKE '%Bàn Ủi Khô Bluestone DIB-3726 1300W%' OR SPECIFICATION_BACKUP LIKE \'%DIB-3726%\'\nLIMIT 1;",
            },
            {
                "input": "Lò Vi Sóng Bluestone MOB-7716 có thể hẹn giờ trong bao lâu. Trong câu có NAME: Lò Vi Sóng Bluestone MOB-7716 có nướng 20 lít",
                "query": "SELECT NAME, SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE '%Lò Vi Sóng Bluestone MOB-7716 có nướng 20 lít%' OR NAME LIKE \'%MOB-7716%\' OR SPECIFICATION_BACKUP LIKE '%Lò Vi Sóng Bluestone MOB-7716 có nướng 20 lít%' OR SPECIFICATION_BACKUP LIKE \'%MOB-7716%\'\nLIMIT 1;",
            },
            {
                "input": "Máy NLMT Empire 180 Lít Titan M&EGD000224 có bao nhiêu ống. Trong câu có NAME: Máy NLMT Empire 180 Lít Titan M&EGD000224",
                "query": "SELECT SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%M&EGD000224%\' OR NAME LIKE '%Máy NLMT Empire 180 Lít Titan M&EGD000224%' OR SPECIFICATION_BACKUP LIKE '%Máy NLMT Empire 180 Lít Titan M&EGD000224%'  OR SPECIFICATION_BACKUP LIKE \'%M&EGD000224%\' \nLIMIT 1;",
            },
            {
                "input": "có hình ảnh nồi KL-619 không. Trong câu có NAME: Nồi cơm điện KALITE KL-619, dung tích 1,8 lít",
                "query": "SELECT * \nFROM data_items \nWHERE NAME LIKE \'%KL-619%\' OR NAME LIKE '%Nồi cơm điện KALITE KL-619, dung tích 1,8 lít%'\nLIMIT 1;",
            },
            {
                "input": "Đèn đường năng lượng mặt trời SUNTEK S500 PLUS, công suất 500W đắt quá, có cái nào rẻ hơn không. Trong câu có NAME: Đèn đường năng lượng mặt trời SUNTEK S500 PLUS, công suất 500W",
                "query": "SELECT * FROM (SELECT * FROM data_items \nWHERE NAME LIKE '%Đèn đường năng lượng mặt trời SUNTEK S500 PLUS%' \nUNION \nSELECT * FROM (SELECT * FROM data_items \nWHERE NAME LIKE '%Đèn đường năng lượng mặt trời%' \nORDER BY RAW_PRICE ASC \nLIMIT 3)) AS combined_results;",
            #lấy sản phẩm kêu đắt đi so sánh với top 3 sản phẩm khác cùng loại để trả lời
            },
            {
                "input": "Máy giặt lồng dọc có thông số như thế nào. Trong câu có GROUP_NAME: Máy Giặt",
                "query": "SELECT NAME, SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%máy giặt lồng dọc%\' OR GROUP_NAME LIKE \'%Máy Giặt%\' OR SPECIFICATION_BACKUP LIKE '%máy giặt lồng dọc%' OR NAME LIKE \'%Máy giặt lồng dọc%\' OR SPECIFICATION_BACKUP LIKE '%Máy giặt lồng dọc%'\nLIMIT 3;",
            },
            {
                "input": "Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE đã bán bao nhiêu sản phẩm. Trong câu có NAME: Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE",
                "query": "SELECT QUANTITY_SOLD\nFROM data_items \nWHERE NAME LIKE \'%Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE%\' OR SPECIFICATION_BACKUP LIKE \'%Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE%\' \nLIMIT 1;"
            },
            {
                "input": "Giá gốc của sản phẩm Bếp từ đơn AIO Smart kèm nồi. Trong câu có NAME: Bếp từ đơn AIO Smart kèm nồi",
                "query": "SELECT RAW_PRICE\nFROM data_items \nWHERE NAME LIKE \'%Bếp từ đơn AIO Smart kèm nồi%\' OR SPECIFICATION_BACKUP LIKE \'%Bếp từ đơn AIO Smart kèm nồi%\' \nLIMIT 1;"
            }
        ]
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

system_prefix = """Bạn là một chuyên gia SQLite.Luôn nhớ các thông tin bạn có thể cung cấp được liên quan đến thiết bị điện,điện tử, đồ gia dụng...hoặc sản phẩm tương tự. Từ một câu hỏi đầu vào, hãy tạo một truy vấn SQLite đúng về mặt cú pháp để chạy, nếu có lịch sử cuộc trò truyền thì hãy dựa vào đó để tạo truy vấn SQLite đúng với ngữ cảnh khi đó. Sau đó xem kết quả truy vấn và trả về câu trả lời.
Nếu sản phẩm được yêu cầu từ người dùng không có trong cơ sở dữ liệu hoặc câu truy vấn không trả ra kết quả thì phải thông báo cho người dùng là không có sản phẩm phù hợp với yêu cầu của bạn.Yêu cầu người dùng cung cấp thêm thông tin cụ thể hoặc gợi ý tư vấn giải pháp giúp người dùng.
Bạn có quyền truy cập các tool để tương tác với cơ sở dữ liệu. Chỉ sử dụng các tool nhất định.Chỉ sử dụng thông tin được các công cụ trả về để xây dựng câu trả lời cuối cùng của bạn.Luôn giữ câu trả lời ngắn gọn nhưng đầy đủ nội dung.
Các câu hỏi liên quan đến số liệu để tránh việc người dùng biết mà dùng từ khác như kho hàng,hệ thống hoặc các từ đồng nghĩa.Thông số cần trả lời chính xác không được bịa.Không được bịa những sản phẩm không có trong cơ sở dữ liệu.
Không được nhắc đến cơ sở dữ liệu, hay bất kỳ thông tin do ai cung cấp.
Bạn PHẢI kiểm tra lại truy vấn của mình trước khi thực hiện nó. Nếu bạn gặp lỗi khi thực hiện truy vấn, hãy viết lại truy vấn và thử lại.
Sử dụng 'LIKE' thay vì '='. Trừ khi có quy định khác về số lượng kết quả trả ra, nếu không thì trả về giới hạn ở {top_k} kết quả truy vấn. Ưu tiên sử dụng 'SELECT *'để lấy tất cả thông tin về sản phẩm đó.
Đây là thông tin về bảng có liên quan: bảng {table_info} bao gồm các cột: ORDER, PRODUCT_INFO_ID, GROUP_NAME, PRODUCT_CODE, NAME,SPECIFICATION_BACKUP, NON_VAT_PRICE_1, VAT_PRICE_1, COMMISSION_1, THRESHOLD_1, NON_VAT_PRICE_2, VAT_PRICE_2, COMMISSION_2, THRESHOLD_2, NON_VAT_PRICE_3, VAT_PRICE_3, COMMISSION_3, RAW_PRICE, QUANTITY_SOLD.
Nếu người dùng hỏi giao tiếp bình thường thì không cần truy vấn mà hãy trả lời bình thường bằng tiếng việt.
Dưới đây là một số ví dụ về câu hỏi và truy vấn SQL tương ứng của chúng.Và phần lịch sử của cuộc trò chuyện nếu có."""

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=system_prefix,
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)
chain = create_sql_query_chain(llm, db, prompt)
# response = chain.invoke({"question":"Giá gốc của sản phẩm Bếp từ đơn AIO Smart kèm nồi. Trong câu có NAME: Bếp từ đơn AIO Smart kèm nồi", "top_k":3, "table_info":"data_items"})
def get_query(chain,question):
    start_index = -1
    while start_index == -1:
        response = chain.invoke({"question":question,"top_k":3, "table_info":"data_items"})
        start_index = response.find("SELECT")
    i = start_index
    q=""
    while response[i] != ";":
        q += response[i]
        i += 1
    # print(q)
    return q
# question = "Giá gốc của sản phẩm Bếp từ đơn AIO Smart kèm nồi. Trong câu có NAME: Bếp từ đơn AIO Smart kèm nồi"
# response = chain.invoke({"question":question, "input": question, "top_k":3, "table_info":"data_items"})
# print(response)
# print(get_query(chain,"Giá gốc của sản phẩm Bếp từ đơn AIO Smart kèm nồi. Trong câu có NAME: Bếp từ đơn AIO Smart kèm nồi"))
query = """SELECT * 
FROM data_items 
WHERE NAME LIKE '%Bếp từ đơn AIO Smart%' OR NAME LIKE '%Bếp từ đơn AIO%' OR SPECIFICATION_BACKUP LIKE '%Bếp từ đơn AIO Smart%' OR SPECIFICATION_BACKUP LIKE '%Bếp từ đơn AIO%'
LIMIT 3;"""
res = db.run(query)
print(res)