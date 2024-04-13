from openai import OpenAI
import streamlit as st
import requests
import json
import os

# Set the title of the Streamlit app
st.title('🩺🏥 Allamat Medical Chatbot')

# Assuming 'openai_api_key' is correctly set in your Streamlit secrets
client = OpenAI(api_key=st.secrets["default"]["openai_api_key"])

# Access your Google API key securely from secrets
google_api_key = st.secrets["default"]["google_api_key"]

# Define the directory and file for storing chat data
CHAT_DATA_DIR = 'chat_data'
CHAT_HISTORY_FILE = 'chat_histories.json'

# Ensure the directory for chat data exists
os.makedirs(CHAT_DATA_DIR, exist_ok=True)

# Function to save chat histories to a JSON file
def save_chat_histories():
    with open(os.path.join(CHAT_DATA_DIR, CHAT_HISTORY_FILE), 'w') as file:
        json.dump(st.session_state.chat_histories, file, indent=4)

# Function to load chat histories from a JSON file
def load_chat_histories():
    try:
        with open(os.path.join(CHAT_DATA_DIR, CHAT_HISTORY_FILE), 'r') as file:
            st.session_state.chat_histories = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize an empty dictionary
        st.session_state.chat_histories = {}

# Load chat histories when the app starts
load_chat_histories()

# Initialize session state for storing chat messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the model type in session state if not present
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Function to manage chat sessions
def manage_chat_sessions():
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    def save_current_session(session_name):
        st.session_state.chat_histories[session_name] = st.session_state.messages.copy()
        save_chat_histories()

    def load_session(session_name):
        st.session_state.messages = st.session_state.chat_histories[session_name].copy()
        save_chat_histories()  # Save after loading a session

    def start_new_session():
        st.session_state.messages = []
        save_chat_histories()  # Save after starting a new session

    st.sidebar.title("Chat Histories")
    session_name = st.sidebar.text_input("Save current session as:", key="save_session_name")
    if st.sidebar.button("Save Session"):
        save_current_session(session_name)
    if st.sidebar.button("New Session"):
        start_new_session()
    selected_session = st.sidebar.selectbox("Select a session to view:", options=list(st.session_state.chat_histories.keys()))
    if st.sidebar.button("Load Session"):
        load_session(selected_session)

# Call manage_chat_sessions function
manage_chat_sessions()

# Function to fetch healthcare places from Google Places API
def fetch_healthcare_places(query, location='Singapore'):
    base_url = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
    params = {
        'query': f"{query} clinic near {location}",
        'key': google_api_key,
        'type': 'doctor|health'
    }
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            results = response.json()['results']
            return [(place['name'], place.get('formatted_address', 'No address provided')) for place in results]
        else:
            print(f"Failed to fetch places: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error during API request: {e}")
        return []

# Function to display a list of clinics in the sidebar
def show_clinics_in_sidebar(recommendations):
    if recommendations:
        st.sidebar.header("Recommended Clinics")
        for name, address in recommendations:
            st.sidebar.text(f"{name}\n{address}\n")

# Display existing messages and handle new user input
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "user":
            recommendations = fetch_healthcare_places(message["content"])
            show_clinics_in_sidebar(recommendations)

# Chat input and response generation
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_chat_histories()  # Save after adding a new message
