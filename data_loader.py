from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def process_pdf(file_path):
    doc_reader = PdfReader(file_path)
    doc_page = []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    for index, page in enumerate(doc_reader.pages):
        page_text = page.extract_text()
        texts = text_splitter.split_text(page_text)
        for paragraph_id, paragraph in enumerate(texts):
            doc_page.append(Document(page_content=paragraph,
                            metadata=dict(paragraph_id=paragraph_id, page=index + 1)))
    return doc_page
