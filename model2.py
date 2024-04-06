from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"
LLM_PATH="mistral-7b-instruct-v0.2.Q3_K_S.gguf" # Adjust the file name and path accordingly

# 快速回复的字典
quick_replies = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello! How can I assist you today?",
    "how are you": "I'm a allamak, so I don't have feelings, but thanks for asking! How can I assist you?",
    "what's your name": "I'm a medical assistant chatbot here to answer your questions. How can I help you today?",
    "thank you": "You're welcome! If you have any more questions, feel free to ask.",
    "bye": "Goodbye! If you have more questions in the future, don't hesitate to reach out.",
    # 更多例子
    "help": "Sure, I can help. What specific question do you have?",
    "who are you": "I am an AI assistant designed to provide information and answer your questions. How may I assist you today?"
    # 根据需要添加更多
}

def get_quick_reply(user_message):
    # 将用户消息转换为小写，并检查是否有对应的快速回复
    user_message = user_message.lower().strip()
    return quick_replies.get(user_message)

def process_user_input(user_input):
    # 首先尝试匹配快速回复
    reply = get_quick_reply(user_input)
    if reply:
        return reply  # 如果找到快速回复，直接返回

    # 如果没有匹配的快速回复，继续执行原有的处理逻辑
    # 这里是您原有的逻辑，可能包括调用模型、查询数据库等
    detailed_reply = your_original_logic(user_input)
    return detailed_reply

# 假设的原有逻辑函数
def your_original_logic(user_input):
    # 处理用户输入，生成回答的逻辑
    return "这里是根据用户输入生成的详细回答。"


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

   llm = LlamaCpp(
      model_path=LLM_PATH,
      temperature=0.75,
      max_tokens=2000,
      top_p=1
      )

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
    # 首先尝试匹配快速回复
    quick_reply = get_quick_reply(message.content)
    if quick_reply:
        # 如果找到快速回复，直接发送回复并返回
        await cl.Message(content=quick_reply).send()
        return

    chain = cl.user_session.get("chain") # https://docs.chainlit.io/concepts/user-session
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens = ["FINAL","ANSWER"]
    )
    cb.answer_reached=True
    res = await chain.ainvoke(message.content, callbacks=[cb]) # https://docs.chainlit.io/examples/qa  |ctrl-F message.content
    answer = res["result"]
    # 已移除sources的添加逻辑

    # 直接发送回答，不包含源信息
    await cl.Message(content=answer).send()
