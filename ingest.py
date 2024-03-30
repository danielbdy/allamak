from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from data import get_data_from_bucket
import os

# Specify the file name you want to retrieve from the bucket
file_name = "71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf"

# Call the function to retrieve data from the bucket
data = get_data_from_bucket(file_name)

# Define the local directory to save the downloaded file
# LOCAL_DATA_PATH = 'raw_data/'


# Define the directory where the vector database will be saved
DB_FAISS_PATH = 'vectorstores/dbfaiss'

def create_vector_db():
    loader = DirectoryLoader('.', glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap =50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings( model_name =  'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device' : 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == '__main__':
    create_vector_db()
