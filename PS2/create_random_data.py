import numpy as np
import pandas as pd

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 生成 370 个节点，每个节点有 99 个特征
num_nodes = 2
num_features = 99
node_features = np.zeros((num_nodes, num_features), dtype=int)
for i in range(num_nodes):
    node_features[i, np.random.randint(num_features)] = 1
# node_features = 2, size = )

# 将节点特征保存为 CSV 文件
node_features_df = pd.DataFrame(node_features, columns=[f'feature{i+1}' for i in range(num_features)])
node_features_df.to_csv('node_features.csv', index=False)

# 生成随机的边信息
num_edges = num_nodes * 5  # 每个节点平均 5 条边
edges = np.random.randint(0, num_nodes, size=(num_edges, 2))

# 确保没有自环（边的起点和终点不能相同）
edges = edges[edges[:, 0] != edges[:, 1]]

# 将边信息保存为 CSV 文件
edges_df = pd.DataFrame(edges, columns=['source', 'target'])
edges_df.to_csv('edges.csv', index=False)