import os
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def setup_chain(docsearch):
    llm = OpenAI(api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory
    )

    return crc, memory
