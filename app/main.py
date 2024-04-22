from data_loader import process_pdf
from model_v2 import setup_chain
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import time


def allamak(query, crc, memory):
    """Function to handle query processing with conversation history."""
    history = memory.buffer
    response = crc({"question": query, "chat_history": history})
    return response['answer']

def main():
    """Initializes and returns the components necessary for the chat system."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    pdf_file_path = '71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'

    # Load and process PDF, then create embeddings
    start_time_pdf = time.time()
    doc_page = process_pdf(pdf_file_path)
    pdf_processing_time = time.time() - start_time_pdf
    start_time_faiss = time.time()
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    docsearch = FAISS.from_documents(documents=doc_page, embedding=embedding)
    faiss_loading_time = time.time() - start_time_faiss




    # Setup the conversational chain with the prepared vector store
    start_time_setup = time.time()
    crc, memory = setup_chain(docsearch)
    setup_time = time.time() - start_time_setup

    print("PDF processing time:", pdf_processing_time)
    print("FAISS loading time:", faiss_loading_time)
    print("Setup time:", setup_time)

    return crc, memory

if __name__ == "__main__":
    crc, memory = main()
