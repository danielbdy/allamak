FROM python:3.10.6-buster
# FROM tensorflow/tensorflow


# WORKDIR /app

COPY app /app
COPY requirements.txt /requirements.txt

# COPY ./vectorstores ./vectorstores
# COPY ./LLM ./LLM

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# CMD ["uvicorn", "app.ok_model2:app", "--host", "0.0.0.0"]
CMD uvicorn app.ok_model2:app --host 0.0.0.0 --port $PORT
# CMD ["streamlit","run", "appy2:app", "--host", "0.0.0.0"]
# CMD ["streamlit", "run", "--server.port", "8501", "appy2.py"]
