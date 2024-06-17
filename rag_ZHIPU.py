import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中读取 ZhipuAI API Key
api_key = os.getenv("ZHIPUAI_API_KEY")

if not api_key:
    raise ValueError("ZHIPUAI_API_KEY 环境变量未设置")

os.environ["ZHIPUAI_API_KEY"] = api_key

# 初始化 FastAPI 应用
app = FastAPI()

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

# 自定义提示模板
def custom_prompt(question):
    return f"""你是一个智能助手，任务是根据提供的问题生成回答。请按照以下格式分点作答：
1. 要点一
2. 要点二
3. 要点三

问题如下：
{question}

请基于上述问题给出你的回答。"""

# 初始化 ZhipuAI 生成模型
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)

# 创建 RAG 链条
def rag_chain(question):
    prompt_text = custom_prompt(question)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt_text),
    ]
    response = chat.invoke(messages)
    return response.content

class QueryModel(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: QueryModel):
    try:
        answer = rag_chain(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
