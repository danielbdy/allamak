from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings

def setup_chain(documents):
    api_key =  #input OpenAI key
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    docsearch = FAISS.from_documents(documents=documents, embedding=embedding)
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
