from config import Config
from ner.ner import NER
from history.history import HistoryProcessor
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
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from config import Config
import os
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
llm = ChatGroq(model="gemma2-9b-it")
db = SQLDatabase.from_uri(Config.db_sql)
model = NER()
question = "2 loại Bếp từ đơn AIO Smart kèm nồi và Bếp từ đôi AIO Smart bếp nào rẻ hơn"
name_entity = model.predict(question)
print(name_entity)
chain = ChainSQL().create_chain(llm,db)
retrieval_doc = DocumentRetrieval()
if len(name_entity['NAME']) <=2 and len(name_entity['NAME']) > 0:
    context = ""
    for item in name_entity['NAME']:
        context += retrieval_doc.retrieve_documents(item) +'\n'
    print(context)
    answer_prommt = ChatPromptTemplate.from_template("""
         Dưới đây là ngữ cảnh:
         {context}
         Hãy trả lời câu hỏi sau dựa trên ngữ cảnh trên:
         Câu hỏi: {question}
         Câu trả lời:                              
         """)
    basic_chain = (
        {
            "context":itemgetter("context"),
            "question":itemgetter("question"),
        }
        | answer_prommt
        | llm
        | StrOutputParser()
    )
    respone = basic_chain.invoke({"context":context,"question":question})
    print(respone)
elif len(name_entity['GROUP_NAME']) > 0:
    pass

