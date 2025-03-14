import os
import time

import pandas as pd
from flask import Flask, request, jsonify
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

model = ChatOpenAI(
    api_key="sk-xxx",
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.01
)
embeddings = OpenAIEmbeddings(model="BAAI/bge-m3",
                              api_key="sk-xxx",
                              base_url="https://api.siliconflow.cn/v1")

ASTRA_DB_API_ENDPOINT = "https://xxx.apps.astra.datastax.com"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:xxx"
# 默认只开一个collection
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    collection_name="astra_vector_langchain",
    token=ASTRA_DB_APPLICATION_TOKEN,
)
app = Flask(__name__)
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


def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


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
    ai_data = get_excel_data(file_path)
    for batch in batch_data(ai_data, 60):
        vector_store.add_texts(batch)
    return {
        "code": 200,
        "data": {"dataset_id": "astra_vector_langchain"},
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
        # dataset_id = data["dataset_id"]
        retrieved_docs = vector_store.similarity_search(data['query'])
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.invoke({"question": data['query'], "context": docs_content})
        print(messages)
        response = model.invoke(messages)
        return jsonify({"answer": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
