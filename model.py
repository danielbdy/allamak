# from langchain_community import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
# from transformers import AutoTokenizer, AutoModelForCausalLM

DB_FAISS_PATH = "vectorstores/dbfaiss"

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

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

    return prompt

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

    # llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='gemma_gpu.bin',local_files_only=True,from_pt=True)
    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt':prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device':'cpu'})

    # db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    # The de-serialization relies loading a pickle file. Pickle files can be modified to deliver a malicious payload that results in
    # execution of arbitrary code on your machine.You will need to set allow_dangerous_deserialization to True to enable deserialization.
    # If you do this, make sure that you trust the source of the data. For example, if you are loading a file that you created, and no that
    # no one else has modified the file, then this is safe to do. Do not set this to True if you are loading a file from an untrusted source
    # (e.g., some random site on the internet.).

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response

### Chainlit ###
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, I am allamak. What is your question?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") # https://docs.chainlit.io/concepts/user-session
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens = ["FINAL","ANSWER"]
    )
    cb.answer_reached=True
    res = await chain.ainvoke(message.content, callbacks=[cb]) # https://docs.chainlit.io/examples/qa  |ctrl-F message.content
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo sources found"

    await cl.Message(content=answer).send()
