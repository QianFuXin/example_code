"""
输入n个句子，计算n个句子之间的相似度，然后通过密度聚类，进行分簇，实现分类的效果。
比如有很多用户评论、聊天记录等发散型数据，可以先用大模型进行意图识别、总结该文本内容，然后通过语义相似度，密度聚类实现自动分类的效果。
---
提示词：
任务说明：
以下是用户与客服的聊天记录，请从用户的发言中提取反馈的核心问题或现象，并将其总结为简洁的一句话，仅返回结果，不包含其他内容。

数据特性：
    •	该聊天记录由语音转文本生成，可能包含错别字或语序混乱。
    •	请结合上下文语义进行分析，不要被错别字或拼写错误误导。
    •	如遇错别字或拼写错误，请合理推测正确拼写并理解原意。
    •	仅关注用户的发言，不要包含客服的回复。

示例：
聊天记录： “为什么xxx”
返回： 用户询问xxx的原因。

聊天记录： {chat_history}
返回：
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

# 加载模型和分词器
model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# 示例输入文本
input_texts = ["你好", "世界", "机器学习", "人工智能", "深度学习", "数据科学", "你好世界", "AI", "大模型", "GPT"]

# 文本向量化
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

with torch.no_grad():
    outputs = model(**batch_dict)

# 获取 CLS token 向量
embeddings = outputs.last_hidden_state[:, 0]
embeddings = F.normalize(embeddings, p=2, dim=1)  # 归一化

# 计算相似度矩阵 [0-1]
similarity_matrix = (embeddings @ embeddings.T).detach().cpu().numpy()
distance_matrix = 1 - similarity_matrix

# 处理浮点误差问题
distance_matrix = np.round(distance_matrix, decimals=6)
distance_matrix = np.clip(distance_matrix, 0, 1)  # 确保所有值在 0-1 之间

# print("相似度矩阵:")
print(distance_matrix)

# 密度聚类
clustering = DBSCAN(eps=0.25, min_samples=2, metric="precomputed").fit(distance_matrix)

# 获取聚类标签
labels = clustering.labels_
print("聚类标签:", labels)

# 输出结果
result = pd.DataFrame({"text": input_texts, "cluster_label": labels})
# print(result)
# 统计各个簇的数量
group_counts = result["cluster_label"].value_counts()
print("\nGroup Counts:")
print(group_counts)

# # 找到最大簇
# most_common_label = group_counts.idxmax()
# print(f"\nMost Common Cluster Label: {most_common_label}")
#
# # 筛选最大簇数据
# filtered_result = result[result["cluster_label"] == most_common_label]
# print("\nFiltered Result:")
# print(filtered_result)
