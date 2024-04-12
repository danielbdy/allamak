FROM python:3.8.6-buster

# WORKDIR /app

COPY app /app
COPY requirements.txt /requirements.txt

# COPY ./vectorstores ./vectorstores
# COPY ./LLM ./LLM

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# CMD ["uvicorn", "app.ok_model2:app", "--host", "0.0.0.0"]
CMD uvicorn app.ok_model2:app --host 0.0.0.0 --port $PORT
