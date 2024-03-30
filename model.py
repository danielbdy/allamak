# from langchain_community import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# from transformers import AutoTokenizer, AutoModelForCausalLM

DB_FAISS_PATH = "vectorstores/dbfaiss"

custom_prompt_template = '''
As a medical chatbot, your goal is to engage in a conversational, empathetic, and informative dialogue. Reflect on the previous conversation to maintain context and continuity. Provide evidence-based medical information in response to the user's question. If the answer is not within your expertise, clearly state your limitation and suggest seeking professional medical advice.

Previous Conversation:
{history}

Based on the context and the user's current question, formulate a response that is both informative and empathetic. Ensure your answer adheres to medical accuracy and current healthcare guidelines.

Context: {context}
User's Question: {question}

Answer thoughtfully, prioritizing the user's need for clear, reliable, and actionable medical advice. If unsure, it's crucial to admit the limitation and guide the user towards professional help.

Helpful answer:
'''

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question', 'history'])
    return prompt

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5,

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
    initial_history = "Hi, I am allamak. How can I assist you today?"
    cl.user_session.set("chain", chain)
    cl.user_session.set("history", initial_history)
    await cl.Message(content=initial_history).send()


@cl.on_message
async def main(message):
    chain = cl.user_session.get('chain')

    # Get existing history or initialize to empty string
    history = cl.user_session.get('history', '')

    # Update history with the new message
    updated_history = f"{history}\nUser: {message.content}"
    cl.user_session.set('history', updated_history)

    # Prepare the input for the chain call, including the updated history
    input_data = {
        'query': message.content,
        'history': updated_history
    }

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens = ["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    print(input_data)

    # Pass the input data to the chain call
    res = await chain.acall(input_data, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    # Append bot's response to the history
    updated_history += f"\nBot: {answer}"
    cl.user_session.set('history', updated_history)

    if sources:
        answer += f"\nSources:" + ', '.join([src['title'] for src in sources])
    else:
        answer += "\nNo Sources Found"
    await cl.Message(content = answer).send()
