# import streamlit as st
# from data_loader import process_pdf
# from model_v2 import setup_chain

# st.title('PDF Document Query System')

# # Path to the PDF file
# pdf_file_path = 'content/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'

# query = st.text_input("Enter your query:")
# if query:
#     with st.spinner('Processing PDF...'):
#         # Process the specified PDF file
#         document_pages = process_pdf(pdf_file_path)
#         crc, memory = setup_chain(document_pages)
#         response = crc({"question": query, "chat_history": memory.buffer})
#         st.write("Response:", response)

# if st.button('Clear History'):
#     memory.clear()
#     st.success('History cleared!')

# app.py
import streamlit as st
from streamlit_chat import message
import main

st.subheader("Allamak - your A-D diagnosis chatbot")

# Initialize the application components once
if 'crc' not in st.session_state or 'memory' not in st.session_state:
    st.session_state.crc, st.session_state.memory = main.main()

# Setup UI and session state for managing conversation
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Welcome! How can I assist you today?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Chat UI
response_container = st.container()
text_container = st.container()

with text_container:
    user_input = st.text_input("Type your question here:", key="input")
    if user_input:
        st.session_state.requests.append(user_input)
        response = main.allamak(user_input, st.session_state.crc, st.session_state.memory)
        st.session_state.responses.append(response)

with response_container:
    for i in range(len(st.session_state['responses'])):
        if i < len(st.session_state['requests']):
            message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['responses'][i], key=str(i))
