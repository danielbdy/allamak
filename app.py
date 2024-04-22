import streamlit as st
from streamlit_chat import message
import main

# Assuming your image is named 'doctor_image.png' and is in the same directory as your script
image_path = 'image/allamak_rebrand.png'

col1, col2 = st.columns([1, 3])

with col1:
    st.image(image_path, width=200)  # Adjust width as needed

with col2:
    st.subheader("Allamak - your personal AI doctor")
    st.markdown("Hello I'm Allamak! We can work on a pressing problem. Just tell me what’s wrong. You can say “I have a sore throat” or \"What is Anemia?\"")
# st.subheader("Allamak - your diagnosis chatbot")

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
