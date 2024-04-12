from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from fastapi.responses import Response
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

DB_FAISS_PATH = "app/vectorstores/db_faiss"
LLM_PATH = "app/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf" # Adjust the file name and path accordingly
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say you don't know the answer. Do not try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    llm = LlamaCpp(model_path=LLM_PATH, temperature=0.75, max_tokens=2000, top_p=1)
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'verbose': True, 'prompt': prompt})
    return qa_chain

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
llm = load_llm()
qa_prompt = set_custom_prompt()
chain = retrieval_qa_chain(llm, qa_prompt, db)

class ChatRequest(BaseModel):
    message: str

async def generate_response(query: str) -> str:
    print(f"Generating response for query: {query}")
    try:
        inputs = {"query": query}
        print(f"Inputs prepared for the chain: {inputs}")
        response = await chain._acall(inputs)
        print(f"Response from chain: {response}")
        answer = response.get("result", "No answer generated.")
        return answer
    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise

@app.post("/chat")
async def chat(request: ChatRequest):
    print(request.dict())
    user_message = request.message
    try:
        response_message = await generate_response(user_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"response": response_message}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the chatbot API!"}
