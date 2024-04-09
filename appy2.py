import streamlit as st
import requests
from datetime import datetime

st.title('Chat with Allamak')

user_input = st.text_input("Message:")

if st.button('Send'):
    if user_input:
        payload = {"message": user_input}
        input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = requests.post("http://localhost:8000/chat", json=payload)

        if response.status_code == 200:
            response_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            bot_response = response.json().get("response", "No response generated by the API.")
            st.text_area("Response: ", value=f"{bot_response}\nMessage sent at: {input_time}\nResponse received at: {response_time}", height=100)
        else:
            # Handling errors more gracefully
            error_details = response.json().get('detail', 'Unknown error')
            st.error(f"Error from the API: {error_details}")
    else:
        st.error("Please enter a message.")