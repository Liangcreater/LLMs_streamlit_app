# 构建strealit_LLM_APP
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
# 设置中国的本地变量
# 设置api
os.environ["OPENAI_API_KEY"] = "sk-VsmUlSeewGfzUJnLB408901cB3Da4f8aA68711735e712aC5"
# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:33210"
os.environ["https_proxy"] = "http://127.0.0.1:33210"
# 设置中转
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"





# 定义一个函数，使用用户密钥对OpenAI API进行身份验证、发送提示并获取AI生成的响应
def generate_response(input_text):
    llm = ChatOpenAI(temperature=0.7)
    return llm.predict(input_text)



# 添加检索问答
# 先将构建检索问答链部分的代码进行封装：
#     get_vectordb函数返回C3部分持久化后的向量知识库
#     get_chat_qa_chain函数返回调用带有历史记录的检索问答链后的结果
#     get_qa_chain函数返回调用不带有历史记录的检索问答链后的结果


# get_vectordb
# 导入相关的库
# 导入embedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# 获取持久化向量知识库的函数
def get_vectordb():
    # 定义 Embeddings
    embedding = OpenAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = './data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb


# get_chat_qa_chain
# 带有历史记录的问答链
# 导入对话检索链
from langchain.chains import ConversationalRetrievalChain
# 导入memory模块
from langchain.memory import ConversationBufferMemory
# 定义获取对话问答链
def get_chat_qa_chain(question:str):
    """
    question: param: 输入的问题prompt
    """
    # 初始化向量数据库
    vectordb = get_vectordb()
    # 初始化llm
    llm = ChatOpenAI(temperature = 0)
    # 初始化memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    # 构造检索器
    retriever=vectordb.as_retriever()
    # 构建qa链
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    # 运行对话
    result = qa({"question": question})
    # 返回结果
    return result['answer']

# get_qa_chain
#不带历史记录的问答链
# 导入提示模板
from langchain.prompts import PromptTemplate
# 导入检索问答链
from langchain.chains import RetrievalQA
# 定义不带历史记录的问答链
def get_qa_chain(question:str):
    """
    question: param: 输入的问题prompt
    """
    # 构造向量数据库
    vectordb = get_vectordb()
    # 构造llm
    llm = ChatOpenAI(temperature = 0)
    # prompt模板
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    # 构造prompt
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    # 实例化问答链
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),   # 构建检索器
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]






# 通过使用 st.session_state 来存储对话历史，可以在用户与应用程序交互时保留整个对话的上下文。

# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 动手学大模型应用开发')
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # 添加一个选择按钮来选择不同的模型
    # 添加一个单选按钮部件st.radio，选择进行问答的模式：
    # None：不使用检索问答的普通模式
    # qa_chain：不带历史记录的检索问答模式
    # chat_qa_chain：带历史记录的检索问答模式
    #selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()