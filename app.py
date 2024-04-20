# import streamlit as st
# from streamlit_chat import message
# import main

# st.subheader("Allamak - your A-D diagnosis chatbot")

# # Initialize the application components once
# if 'crc' not in st.session_state or 'memory' not in st.session_state:
#     st.session_state.crc, st.session_state.memory = main.main()

# # Initialize chat state
# if 'init' not in st.session_state:
#     st.session_state['responses'] = []
#     st.session_state['requests'] = []
#     st.session_state['init'] = True  # Ensures that initialization doesn't happen again

# # Chat UI
# response_container = st.container()
# text_container = st.container()

# with text_container:
#     user_input = st.text_input("Type your question here:", key="input")
#     if user_input:
#         st.session_state.requests.append(user_input)
#         response = main.allamak(user_input, st.session_state.crc, st.session_state.memory)
#         st.session_state.responses.append(response)

# with response_container:
#     # Loop through the number of messages based on the longest list (requests or responses)
#     for i in range(max(len(st.session_state['requests']), len(st.session_state['responses']))):
#         if i < len(st.session_state['requests']):
#             message(st.session_state['requests'][i], is_user=True, key=f"{i}_user")
#         if i < len(st.session_state['responses']):
#             message(st.session_state['responses'][i], key=f"{i}")


import streamlit as st
from streamlit_chat import message
import main

st.subheader("Allamak - your A-D diagnosis chatbot")

# Initialize the application components once
if 'crc' not in st.session_state or 'memory' not in st.session_state:
    st.session_state.crc, st.session_state.memory = main.main()

# Initialize chat state
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'requests' not in st.session_state:
    st.session_state.requests = []
if 'widget' not in st.session_state:
    st.session_state.widget = ''

def submit():
    st.session_state.requests.append(st.session_state.widget)
    response = main.allamak(st.session_state.widget, st.session_state.crc, st.session_state.memory)
    st.session_state.responses.append(response)
    st.session_state.widget = ''  # Clear the widget value after submission

# Display chat messages
for i in range(max(len(st.session_state['requests']), len(st.session_state['responses']))):
    if i < len(st.session_state['requests']):
        message(st.session_state['requests'][i], is_user=True, key=f"{i}_user")
    if i < len(st.session_state['responses']):
        message(st.session_state['responses'][i], key=f"{i}")

# Text input for user query (placed at the bottom)
st.text_input('Type your question here:', key='widget', on_change=submit)
