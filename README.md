# Retrieval Augmented Generation Chatbot Diagnosis System

This project develops a chatbot that leverages Natural Language Processing (NLP) methods and retrieval-augmented generation techniques using OpenAI's Large Language Models. It aims to provide preliminary medical diagnoses based on user-input symptoms.

## Project Description

The Retrieval Augmented Generation (RAG) Chatbot Diagnosis System uses a combination of OpenAI's advanced language models and retrieval-augmented techniques to enhance the chatbot's ability to understand and process medical queries. This system asks relevant questions based on initial symptoms described by the user and guides them towards potential medical advice or diagnosis.

## Features

- **Symptom Analysis**: Analyzes initial user inputs to determine related illnesses.
- **Contextual knowledge**: Understand and has memory of previous conversations allowing for follow up queries
- **Retrieval Augmented Generation**: Enhances LLM's capabilities with other sources of documents allowing for a more comprehensive and attributable answering.
- **Speed Efficiency**: Uses FAISS vectorstore for efficient similarity retrieval in large amount of text agains user queries.

## Installation
pip install -r [v3]requirements.txt
Open AI API required

## Run
streamlit run app.py
