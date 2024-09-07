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
from langchain_core.prompts import (
    ChatPromptTemplate
)
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from config import Config
import os
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
def complete_respone(response : str):
    return response.replace("**",'').strip()
llm = ChatGroq(model="gemma2-9b-it")
db = SQLDatabase.from_uri(Config.db_sql)
model = NER()
chain = ChainSQL().create_chain(llm,db)
retrieval_doc = DocumentRetrieval()

def answer(question:str,id_user:str,id_conversation:str, llm, model, chain, retrieval_doc):
    name_entity = model.predict(question)
    print(name_entity)
    chat_history=HistoryProcessor().load_history(id_user,id_conversation)
    format_chat_his=[]
    for i in range(max(0, len(chat_history) - 5),len(chat_history)):
        format_chat_his.append(HumanMessage(content=chat_history[i]["HumanMessage"]))
        format_chat_his.append(AIMessage(content=chat_history[i]["AIMessage"]))
    respone = ""
    if len(name_entity['NAME']) <=2 and len(name_entity['NAME']) > 0:
        context = ""
        for item in name_entity['NAME']:
            context += retrieval_doc.retrieve_documents(item) +'\n'
        # print(context)
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
            | RunnableLambda(complete_respone)
        )
        respone = basic_chain.invoke({"context":context,"question":question})
        # print(respone)
    elif len(name_entity['GROUP_NAME']) > 0:
        print(2)
        new_question = question + " .Trong câu có GROUP_NAME: "
        for item in name_entity['GROUP_NAME']:
            new_question += item + " "
        if len(name_entity['NAME']) > 0:
            new_question += " .Trong câu có NAME: "
            for item in name_entity['NAME']:
                new_question += item + " "
        print(new_question)
        respone = chain.invoke({"question":new_question, "input": new_question, "top_k":3, "table_info":"data_items","history":format_chat_his})
    elif len(name_entity['GROUP_NAME']) == 0 and len(name_entity['NAME']) == 0:
        respone = chain.invoke({"question":question, "input": question, "top_k":3, "table_info":"data_items","history":format_chat_his})

    HistoryProcessor().update_history(id_user,id_conversation,HumanMessage=question,AIMessage=respone)
    return respone
# id_user = "1"
# id_conversation = "1"
# question = "có bao nhiêu Đèn Năng Lượng Mặt Trời"
# kq = answer(question,id_user,id_conversation,llm,model,chain,retrieval_doc)
# print(kq)