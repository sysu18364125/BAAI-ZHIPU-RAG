import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# 设置 ZhipuAI API Key
os.environ["ZHIPUAI_API_KEY"] = "6e6ed0250d64a8eb9b9eca14603fc4e6.tjS2gPaEz1RZYc7q"

# 初始化 HuggingFaceBgeEmbeddings 模型
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

# 加载、切分和索引博客内容
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 创建向量存储并使用 HuggingFaceBgeEmbeddings 进行嵌入生成
vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings)

# 检索和生成使用博客的相关片段
retriever = vectorstore.as_retriever()

# 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 自定义提示模板
def custom_prompt(context, question):
    return f"""你是一个智能助手，任务是根据提供的文档内容回答问题。请按照以下格式分点作答：
1. 要点一
2. 要点二
3. 要点三

上下文内容如下：
{context}

问题如下：
{question}

请基于上述内容给出你的回答。"""

# 初始化 ZhipuAI 生成模型
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)

# 创建 RAG 链条
def rag_chain(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    context = format_docs(retrieved_docs)
    prompt_text = custom_prompt(context, question)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt_text),
    ]
    response = chat.invoke(messages)
    return response.content

# 执行查询
response = rag_chain("任务分解是什么？")
print("Answer:", response)
