import json
import os
import time

import pandas as pd
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
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

es = Elasticsearch(
    "http://127.0.0.1:9200",
    basic_auth=("elastic", "qianfuxin")
)
ASTRA_DB_API_ENDPOINT = "https://xxx.apps.astra.datastax.com"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:xxx"
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


def rerank_documents(query, documents, model="BAAI/bge-reranker-v2-m3", top_n=4, return_documents=False,
                     max_chunks_per_doc=1024, overlap_tokens=80):
    url = "https://api.siliconflow.cn/v1/rerank"
    headers = {
        "Authorization": f"Bearer sk-cguojatbpogqwpaxfdshyedakqihbwtvqrvgocrpttfaxlhg",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": return_documents,
        "max_chunks_per_doc": max_chunks_per_doc,
        "overlap_tokens": overlap_tokens
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        results = response.json().get("results", [])
        # 根据相关性分数排序，并取前 top_n 个
        sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]
        # 返回文档内容
        return [(documents[item["index"]]) for item in sorted_results]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {
            "code": 400,
            "data": "",
            "msg": f"无文件"
        }
    dataset_id = str(int(time.time()))
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name=dataset_id,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    ai_data = get_excel_data(file_path)
    for batch in batch_data(ai_data, 60):
        vector_store.add_texts(batch)
    # 关键字方式处理数据，存入es
    if not es.indices.exists(index=dataset_id):
        es.indices.create(index=dataset_id)
    bulk(es, [{"_index": dataset_id, "_source": {"content": i}} for i in ai_data])
    return {
        "code": 200,
        "data": {"dataset_id": dataset_id},
        "msg": f"success"
    }


@app.route('/query', methods=['POST'])
def query_answer():
    data = request.json
    if not data or 'dataset_id' not in data or 'query' not in data:
        return {
            "code": 400,
            "data": "",
            "msg": f"缺少dataset_id或query"
        }
    try:
        dataset_id = data["dataset_id"]
        query = data['query']
        vector_store = AstraDBVectorStore(
            embedding=embeddings,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            collection_name=dataset_id,
            token=ASTRA_DB_APPLICATION_TOKEN,
        )
        retrieved_data = [i.page_content for i in vector_store.similarity_search(query, k=4)]
        # 添加关键字匹配
        search_query = {
            "query": {
                "match": {
                    "content": query
                },
            },
            "size": 4
        }
        search_data = [i["_source"]["content"] for i in es.search(index=dataset_id, body=search_query)["hits"]["hits"]]
        # 合并语义检索和关键字检索
        all_data = list(set(retrieved_data + search_data))
        # 重排序
        rerank_data = rerank_documents(query, all_data, top_n=4)
        docs_content = "\n".join(i for i in rerank_data)
        messages = prompt.invoke({"question": query, "context": docs_content})
        print(messages)
        response = model.invoke(messages)
        return jsonify({"answer": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
