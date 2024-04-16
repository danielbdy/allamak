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
    pdf_file_path = 'content/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'

    # Load and process PDF, then create embeddings
    doc_page = process_pdf(pdf_file_path)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    docsearch = FAISS.from_documents(documents=doc_page, embedding=embedding)

    # Setup the conversational chain with the prepared vector store
    crc, memory = setup_chain(docsearch)
    return crc, memory

if __name__ == "__main__":
    crc, memory = main()




main.py
from data_loader import process_pdf
from model_v2 import setup_chain
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pathlib

def allamak(query, crc, memory):
    """Function to handle query processing with conversation history."""
    history = memory.buffer
    response = crc({"question": query, "chat_history": history})
    return response['answer']

def main():
    """Initializes and returns the components necessary for the chat system."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    pdf_file_path = 'content/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'
    embeddings_dir = "embeddings"  # Directory to store the FAISS index

    # Check if embeddings directory exists, create if not
    pathlib.Path(embeddings_dir).mkdir(parents=True, exist_ok=True)

    # Define the path for the FAISS vector store
    db_faiss_path = os.path.join(embeddings_dir, "medical_embeddings.pkl")

    if os.path.isfile(db_faiss_path):
        # Load existing FAISS index
        print("Loading existing FAISS index...")
        embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        docsearch = FAISS.load_local(db_faiss_path, embeddings=embedding,
                                     allow_dangerous_deserialization=True)
    else:
        # Create new FAISS index
        print("Creating new FAISS index...")

        # Load and process PDF, then create embeddings
        doc_page = process_pdf(pdf_file_path)
        embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        docsearch = FAISS.from_documents(documents=doc_page, embedding=embedding)

        # Save the FAISS index
        docsearch.save_local(db_faiss_path)

    # Setup the conversational chain with the prepared vector store
    crc, memory = setup_chain(docsearch)
    return crc, memory

if __name__ == "__main__":
    crc, memory = main()


# main.py
# # main.py
# import os
# from dotenv import load_dotenv
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# import pathlib

# from data_loader import process_pdf
# from model_v2 import setup_chain

# def allamak(query, crc, memory):
#     """Function to handle query processing with conversation history."""
#     history = memory.buffer
#     response = crc({"question": query, "chat_history": history})
#     return response['answer']

# def main():
#     """Initializes and returns the components necessary for the chat system."""
#     load_dotenv()
#     api_key = os.getenv('OPENAI_API_KEY')

#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable not found. Please set it.")

#     pdf_file_path = 'content/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'
#     embeddings_dir = "embeddings"  # Directory to store the FAISS index
#     db_faiss_path = os.path.join(embeddings_dir, "index.pkl")

#     # Create embeddings directory if it doesn't exist
#     os.makedirs(embeddings_dir, exist_ok=True)

#     try:
#         # Process PDF, create embeddings, and attempt to store FAISS index
#         print("Processing PDF and creating embeddings...")
#         doc_page = process_pdf(pdf_file_path)
#         embedding = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
#         docsearch = FAISS.from_documents(documents=doc_page, embedding=embedding)
#         docsearch.save_local(db_faiss_path)
#         print("Embeddings created and index saved.")

#     except Exception as e:
#         print(f"Error during embedding creation: {e}")

#     try:
#         # Load existing FAISS index (even if creation failed)
#         print("Loading FAISS index...")
#         embedding = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
#         docsearch = FAISS.load_local(db_faiss_path, embeddings=embedding)
#         print("Index loaded successfully.")

#     except Exception as e:
#         print(f"Error loading FAISS index: {e}")
#         raise

#     # Setup the conversational chain with the prepared vector store
#     print("Setting up conversational chain...")
#     crc, memory = setup_chain(docsearch)
#     print("Setup complete!")

#     return crc, memory

# if __name__ == "__main__":
#     crc, memory = main()

#     # Add your code to interact with the 'crc' object and handle user queries here
#     # Example:
#     while True:
#         user_input = input("Your question: ")
#         response = allamak(user_input, crc, memory)
#         print("Answer: ", response)
