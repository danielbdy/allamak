from data_loader import process_pdf
from model2 import setup_chain

def allamak(query, crc, memory):
    history = memory.buffer
    response = crc({"question": query, "chat_history": history})
    return response

def main():
    pdf_file_path = 'content/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'
    doc_page = process_pdf(pdf_file_path)
    crc, memory = setup_chain(doc_page)


if __name__ == "__main__":
    main()
