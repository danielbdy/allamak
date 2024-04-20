
from data_loader import process_pdf
from model_v2 import setup_chain
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pathlib


doctor_prompt = """
**You are a helpful medical chatbot. Do not attempt to provide a final diagnosis; that is a doctor's responsibility.  Instead, assist the user by asking questions to gather information about their symptoms and narrow down potential causes. Provide a list of possible relevant questions. Here's how to start:**

**User:** I have a persistent cough.
**You:**  How long have you had the cough? Is it a dry cough or do you produce phlegm? Are there any other symptoms present, such as fever, chest pain, or difficulty breathing?
"""

def allamak(query, crc, memory):
    """Function to handle query processing with conversation history."""
    history = memory.buffer
    # Extract text from HumanMessage objects
    history_text = [message.content for message in history]
    combined_history = doctor_prompt + "\n".join(history_text)
    response = crc({"question": query, "chat_history": combined_history})
    return response['answer']


def main():
    """Initializes and returns the components necessary for the chat system."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    pdf_file_path = 'content/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'
    embeddings_dir = "embeddings"  # Directory to store the FAISS index
    db_faiss_path = os.path.join(embeddings_dir, "index.pkl")

    # Ensure existence of embeddings directory
    pathlib.Path(embeddings_dir).mkdir(parents=True, exist_ok=True)

    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    # Load existing FAISS index (if it exists)
    if os.path.exists(db_faiss_path):
        print("Loading existing FAISS index...")
        docsearch = FAISS.load_local(db_faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        print("FAISS index not found. Creating embeddings...")
        # Load and process PDF, then create embeddings
        doc_page = process_pdf(pdf_file_path)

        # Create new FAISS index (assuming your chunking is done within process_pdf)
        docsearch = FAISS.from_documents(doc_page, embedding)

        # Save the FAISS index
        docsearch.save_local(db_faiss_path)

    # Setup the conversational chain with the prepared vector store
    crc, memory = setup_chain(docsearch)
    return crc, memory

if __name__ == "__main__":
    crc, memory = main()
