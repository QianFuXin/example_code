from fuzzywuzzy import fuzz
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

# 示例数据
data = [
    # 主题 1: 电信运营商相关投诉
    "用户反馈自己申请了宽带注销，但工作人员称手机号仍在最低消费期，无法办理注销。用户表示当初办理时未被告知此条款，认为运营商有欺骗消费者的嫌疑。希望官方能够解释清楚，并给予合理的解决方案，否则会向相关监管机构举报。",

    "用户表示近期话费账单异常，发现过去三个月有额外的增值服务扣费，但本人并未主动开通任何额外服务。联系客服后，客服无法提供具体的开通记录，仅表示‘系统自动开通’，用户怀疑存在恶意扣费，并要求全额退款，否则将向消费者保护协会投诉。",

    "用户反映手机流量套餐自动升级，每月多扣 30 元，联系客服后对方表示无法恢复原套餐，用户对此表示极为不满。用户要求恢复原套餐并退还多扣费用，否则将考虑更换运营商，并在社交媒体上曝光该问题。",

    "客户投诉网络质量极差，家中宽带经常掉线，导致远程办公受到严重影响。多次拨打客服电话后，工程师承诺上门检修，但始终未安排时间。用户认为运营商的服务不到位，并要求免除部分宽带费用，或提供更稳定的网络服务。",

    # 主题 2: 电商购物相关投诉
    "用户在某电商平台购买了一款高端手机，但收到货后发现是翻新机，联系客服后对方要求提供多种证明，并表示‘无法退款’。用户认为平台监管不力，导致假冒伪劣商品流入市场，要求立即全额退款并进行赔偿，否则将向市场监管局举报该平台。",

    "消费者投诉自己在双十一期间购买的商品迟迟未发货，联系客服后仅收到‘系统异常’的回复，但未提供具体解决方案。用户表示，若不能在 3 天内发货，将向消费者保护协会投诉，并要求赔偿因延迟发货造成的损失。",

    "用户申诉自己在电商平台申请售后，因产品质量问题希望退货退款。但商家一直拖延处理，甚至以‘已拆封无法退货’为由拒绝受理。用户认为这是霸王条款，并希望平台介入处理，维护消费者权益。",

    # 主题 3: 银行金融相关投诉
    "用户发现信用卡账单存在异常，某笔交易金额被多次扣款，联系客服后对方表示‘系统问题’，但无法提供退款时间。用户对此感到不满，认为银行服务不到位，并要求立即处理，否则将向银监会投诉。",

    "用户投诉贷款利率突然上调，导致还款压力增加。银行在未提前通知的情况下调整利率，严重影响个人财务规划。用户要求银行提供合理解释，并希望能恢复原利率，否则考虑提前结清贷款并更换金融机构。",

    "用户表示多次尝试联系银行客服处理信用卡冻结问题，但人工客服始终无法接通，仅提供语音机器人应答，且无有效解决方案。用户认为银行客服严重缺失，已在社交平台曝光，并希望银行尽快提供人工服务，否则将向金融监管机构投诉。"
]

# 计算相似度矩阵
n = len(data)
similarity_matrix = np.array([
    [fuzz.token_set_ratio(data[i], data[j]) / 100 if i != j else 1 for j in range(n)]
    for i in range(n)
])

# 计算距离矩阵 (1 - 相似度)
distance_matrix = 1 - similarity_matrix
print(distance_matrix)
# 自动调整 eps（选取 90% 距离作为阈值）
eps_value = np.percentile(distance_matrix, 60)
print(eps_value)
# 进行 DBSCAN 聚类
clustering = DBSCAN(eps=eps_value, min_samples=2, metric="precomputed")
labels = clustering.fit_predict(distance_matrix)

# 统计聚类结果
result_df = pd.DataFrame({"text": data, "cluster_label": labels})

# 统计簇的大小
cluster_counts = result_df["cluster_label"].value_counts()

# 输出聚类结果
print("\n=== 聚类结果 ===")
clusters = {}
for idx, label in enumerate(labels):
    clusters.setdefault(label, []).append(data[idx])

for cluster_id, texts in clusters.items():
    print(f"\n簇 {cluster_id} ({len(texts)} 条文本):")
    for text in texts:
        print(f"  - {text}")

# 统计噪声点
noise_count = sum(labels == -1)
print(f"\n=== 统计信息 ===\n总文本数: {len(data)}")
print(f"噪声点 (-1) 数量: {noise_count} ({noise_count / len(data) * 100:.1f}%)")
print("每个簇的文本数量:\n", cluster_counts)
