FROM python:3.10.6-buster

WORKDIR /streamlit

COPY appy2.py /streamlit/appy2.py
COPY requirements.txt /streamlit/requirements.txt

# COPY ./vectorstores ./vectorstores
# COPY ./LLM ./LLM

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501
# CMD ["streamlit", "run" ,"appy2.py"]
# CMD uvicorn app.ok_model2:app --host 0.0.0.0 --port $PORT
# ENTRYPOINT ["streamlit", "run", "appy2.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD streamlit run appy2.py
# CMD ["streamlit","run", "appy2.py", "--host", "0.0.0.0", "--port", "8501"]
# CMD ["streamlit", "run", "--server.port", "8501", "appy2.py"]
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "appy2.py", "--server.port=8501", "--server.address=0.0.0.0"]
