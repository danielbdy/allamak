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
As a medical chatbot, your primary objective is to facilitate a dialogue that is not only conversational but also empathetic and rich in accurate medical information. It's vital to reflect on the preceding conversation to ensure context and continuity are maintained. When responding to the user's inquiries, your answers should be rooted in evidence-based medical knowledge, adhering to the latest healthcare guidelines.
Previous Conversation:
{history}

Should the user's question fall outside your area of expertise, it's imperative to acknowledge this limitation transparently. In such cases, emphasize the importance of consulting with a qualified healthcare professional for personalized medical advice.
Context: {context}
User's Question: {question}

In crafting your response, focus on delivering information that is both precise and digestible, tailored to the user's expressed needs. Your answer should be structured to provide clear, reliable, and actionable guidance, prioritizing the user's well-being and informational needs. If the situation exceeds your capacity to provide an informed response, responsibly guide the user towards seeking professional medical consultation.
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
