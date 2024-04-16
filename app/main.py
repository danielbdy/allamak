# from data_loader import process_pdf
# from model_v2 import setup_chain
# import os
# from langchain.llms import OpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings.openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# def allamak(query, crc, memory):
#     history = memory.buffer
#     response = crc({"question": query, "chat_history": history})
#     return response

# def main():
#     api_key =  os.getenv('OPENAI_API_KEY')
#     pdf_file_path = 'content/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'
#     doc_page = process_pdf(pdf_file_path)
#     embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
#     docsearch = FAISS.from_documents(documents=doc_page, embedding=embedding)
#     crc, memory = setup_chain(docsearch)


# if __name__ == "__main__":
#     main()

# main.py
# main.py
from data_loader import process_pdf
from model_v2 import setup_chain
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def allamak(query, crc, memory):
    """Function to handle query processing with conversation history."""
    history = memory.buffer
    response = crc({"question": query, "chat_history": history})
    return response['answer']

def main():
    """Initializes and returns the components necessary for the chat system."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    pdf_file_path = 'app/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'

    # Load and process PDF, then create embeddings
    doc_page = process_pdf(pdf_file_path)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    docsearch = FAISS.from_documents(documents=doc_page, embedding=embedding)

    # Setup the conversational chain with the prepared vector store
    crc, memory = setup_chain(docsearch)
    return crc, memory

if __name__ == "__main__":
    crc, memory = main()
