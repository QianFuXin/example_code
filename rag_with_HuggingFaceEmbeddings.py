import os
import time

import pandas as pd
import torch
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

app = Flask(__name__)
VECTOR_DB_PATH = "faiss_data"

llm = ChatOpenAI(
    api_key="ollama",
    model="llama3-CN",
    base_url="http://127.0.0.1:11434/v1",
    temperature=0.01
)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
# BAAI/bge-base-zh-v1.5
model_name = "./model"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
prompt = PromptTemplate.from_template(
    """
    你是一名专注于问答任务的助手。请根据以下提供的上下文内容回答问题。
    如果无法从上下文中获得答案，请明确表示你不知道。
    回答应简洁明了，最多不超过三句话。

    问题： {question}

    上下文： {context}

    回答：
    """
)


def get_excel_data(file_path):
    df = pd.read_excel(file_path, dtype=str)
    return df.apply(lambda row: "|".join([f"{col}:{row[col]}" for col in df.columns]), axis=1).tolist()


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {
            "code": 400,
            "data": "",
            "msg": f"无文件"
        }
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    # 这个地方需要对文件进行类别处理，不同类型的文件，最后都返回数组字符串
    ai_data = get_excel_data(file_path)
    vector_store = FAISS.from_texts(ai_data, embeddings)
    dataset_id = str(int(time.time()))
    vector_store.save_local(os.path.join(VECTOR_DB_PATH, dataset_id))
    return {
        "code": 200,
        "data": {"dataset_id": dataset_id},
        "msg": f"success"
    }


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'dataset_id' not in data or 'query' not in data:
        return {
            "code": 400,
            "data": "",
            "msg": f"缺少dataset_id或query"
        }
    try:
        dataset_id = data["dataset_id"]
        # 如果dataset_idc不存在
        if not os.path.exists(os.path.join(VECTOR_DB_PATH, dataset_id)):
            return {
                "code": 400,
                "data": "",
                "msg": f"dataset_id不存在"
            }
        vector_store = FAISS.load_local(os.path.join(VECTOR_DB_PATH, dataset_id), embeddings,
                                        allow_dangerous_deserialization=True)
        retrieved_docs = vector_store.similarity_search(data['query'])
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.invoke({"question": data['query'], "context": docs_content})
        print(messages)
        response = llm.invoke(messages)
        return jsonify({"answer": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)

# requirements.txt
"""
aiohappyeyeballs==2.5.0
aiohttp==3.11.13
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.8.0
async-timeout==4.0.3
attrs==25.1.0
blinker==1.9.0
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
dataclasses-json==0.6.7
distro==1.9.0
et_xmlfile==2.0.0
exceptiongroup==1.2.2
faiss-cpu==1.10.0
filelock==3.17.0
Flask==3.1.0
frozenlist==1.5.0
fsspec==2025.3.0
greenlet==3.1.1
grpcio==1.67.1
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
httpx-sse==0.4.0
huggingface-hub==0.29.2
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.9.0
joblib==1.4.2
jsonpatch==1.33
jsonpointer==3.0.0
langchain==0.3.20
langchain-community==0.3.19
langchain-core==0.3.43
langchain-huggingface==0.1.2
langchain-milvus==0.1.8
langchain-ollama==0.2.3
langchain-openai==0.3.8
langchain-text-splitters==0.3.6
langgraph==0.3.5
langgraph-checkpoint==2.0.18
langgraph-prebuilt==0.1.2
langgraph-sdk==0.1.55
langsmith==0.3.13
MarkupSafe==3.0.2
marshmallow==3.26.1
milvus-lite==2.4.11
mpmath==1.3.0
msgpack==1.1.0
multidict==6.1.0
mypy-extensions==1.0.0
networkx==3.4.2
numpy==2.2.3
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
ollama==0.4.7
openai==1.65.5
openpyxl==3.1.5
orjson==3.10.15
packaging==24.2
pandas==2.2.3
pillow==11.1.0
propcache==0.3.0
protobuf==6.30.0
pydantic==2.10.6
pydantic-settings==2.8.1
pydantic_core==2.27.2
pymilvus==2.5.5
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
pytz==2025.1
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
requests-toolbelt==1.0.0
safetensors==0.5.3
scikit-learn==1.6.1
scipy==1.15.2
sentence-transformers==3.4.1
six==1.17.0
sniffio==1.3.1
socksio==1.0.0
SQLAlchemy==2.0.38
sympy==1.13.1
tenacity==9.0.0
threadpoolctl==3.5.0
tiktoken==0.9.0
tokenizers==0.21.0
torch==2.6.0
tqdm==4.67.1
transformers==4.49.0
triton==3.2.0
typing-inspect==0.9.0
typing_extensions==4.12.2
tzdata==2025.1
ujson==5.10.0
urllib3==2.3.0
Werkzeug==3.1.3
yarl==1.18.3
zstandard==0.23.0
"""

# Dockerfile
"""
# 使用官方Python基础镜像
FROM python:3.10-slim

# 设置工作目录.dockerignore
WORKDIR /app

# 复制项目的requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 复制应用代码到容器
COPY . .

# 开放应用端口
EXPOSE 5000

# 启动Flask应用
CMD ["python","main.py"]
"""
