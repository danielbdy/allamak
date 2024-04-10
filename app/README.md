# Docker containers and deployment

## 1. LLM and Vectorstores
If the two folders are not shown after you pulled, just create them and place the files in them. See structure below.

**Structure**
```
app/
├── LLM/
│ └── MODEL.gguf
└── vectorstores/
└── db_faiss/
├── index.faiss
└── index.pkl
```

### 1a. LLM portion to update in ok_model2.py
If a different LLM is used, remember to update line 14 of `ok_model2.py` accordingly.
`LLM_PATH = "app/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf"`

For clarity, if the name of the LLM file placed in the LLM folder is gemma-2b-it.gguf, line 14 should be:
`LLM_PATH = "app/LLM/gemma-2b-it.gguf"`

Please note that `ok_model2.py` is using Llama.cpp. The LLM file format should be GGUF.

I have also experimented with the following:
|       LLM       |               Findings               |
|:---------------:|:------------------------------------:|
| gemma-2b-it.gguf | Best performing but too big. To download this file, see previously sent WhatsApp message. |
| gemma-7b-it-Q4_K_M.gguf | From [here](https://huggingface.co/rahuldshetty/gemma-7b-it-gguf-quantized/blob/main/gemma-7b-it-Q4_K_M.gguf). Performance not as good|
| llama-2-7b-chat.ggmlv3.q8_0.bin | Does not work because it is GGML|

Fyi, the mistral-7b-instruct-v0.2.Q4_K_M.gguf can be obtained on [huggingface](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)

### 1b. Vectorstores
You can reuse the previously generated vectorstores.

## 2. Docker container

### 2a. Preparing the docker
If you are testing the docker locally on your machine (e.g., laptop), please uncomment line 14 of the docker file and comment out line 15.

For clarity, lines 14 and 15 of the docker file should look like this.
```dockerfile
# Line 14:
CMD ["uvicorn", "app.ok_model2:app", "--host", "0.0.0.0"]

# Line 15
# CMD uvicorn app.ok_model2:app --host 0.0.0.0 --port $PORT
```

Conversely, if the docker is to be deployed onto GCP, uncomment line 15 and comment out line 14.

### 2b. Creating a docker for testing locally
Start up your docker. Then, in your terminal, enter `docker build -t NAME_OF_DOCKER .`

Once that is done, enter `docker run -p 8000:8000 NAME_OF_DOCKER` into the terminal.

Let's call this terminal, **TERMINAL_A**, for easy identification here.

### 2c. appy2.py
Make sure line 13 of appy2.py reads `response = requests.post("http://localhost:8000/chat", json=payload)`.

Next, open a new terminal (do not close **TERMINAL_A**). For convenience, the new terminal shall be referenced as **TERMINAL_B**.

In **TERMINAL_B**, enter this `streamlit run appy2.py`.

It will open a new browser. From my experience, it opened a chrome but does not load. To resolve this, I copied the URL from the chrome (i.e., http://localhost:8501/), close the chrome browser (this is needed to free up the port), and paste the URL into mozilla firefox.

If you encounter issues, your port might have been taken up by another application. In that case, close all the terminals and start over wth changes to steps 2b and 2c. Change the port for e.g., `docker run -p 8888:8000 NAME_OF_DOCKER` and thus `http://localhost:8888/chat`.

Also, for easy reference, I'm pasting the definitions of the time statistics shown in the terminal by llama.cpp here.
```
load time: time it takes for the model to load.
sample time: time it takes to "tokenize" (sample) the prompt message for it to be processed by the program.
prompt eval time: time it takes to process the tokenized prompt message. If this isn't done, there would be no context for the model to know what token to predict next.
eval time: time needed to generate all tokens as the response to the prompt (excludes all pre-processing time, and it only measures the time since it starts outputting tokens).
```

To close the applications, press ctrl-c in the respective terminals.

## 3. Creating a docker for pushing into GCP
Check your project_id in your GCP. Replace the XXXX in the subsequent line with your project_id.

Enter `export PROJECT_ID="XXXX"` into your terminal. Then, enter `export DOCKER_IMAGE_NAME="allamak-chatbot"` into your terminal.

In the dockerfile, comment out line 14 and uncomment line 15 (see step 2a). Enter `docker build -t eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME .` into the terminal.

After that is completed, enter `docker push eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME`. If you encounter issues here, reach out to the instructors.

Lastly, enter `gcloud run deploy --image eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME --platform managed --region europe-west1`. Similarly, if you encounter issues here, seek assistance from the instructors.

## 4. Docker container deployed
Once the docker has been deployed, you will find a link similar to the one shared in the WhatsApp groupchat. If unsure, ask the instructors.

Now you can connect to your LLM on GCP, update line 13 of appy2.py. See step 2c.

# ~Have fun
