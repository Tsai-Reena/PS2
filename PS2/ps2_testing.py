import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 確保安裝了torch_geometric
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import json

drink_dict = {}

# 讀取 JSON 文件並將其內容存儲到字典中
with open('labels.json', 'r', encoding='utf-8') as fp:
    drink_dict = json.load(fp)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

# 讀取權重文件
weights_path = "./weight/l3bl-GCN_GCN_Custom_arch-3-256-0.5_3-lin3_lr0.01-tem0.07_hidd32-32_multi_model_1.pth"
model_weights = torch.load(weights_path)

# 移除不需要的鍵
keys_to_remove = ["lins.0.weight", "lins.0.bias", "lins.1.weight", "lins.1.bias", "lins.2.weight", "lins.2.bias"]
for key in keys_to_remove:
    if key in model_weights:
        del model_weights[key]

# 讀取節點特徵
node_features = pd.read_csv('node_features.csv').values
X = torch.tensor(node_features, dtype=torch.float32)

# 讀取邊界訊息
edges = pd.read_csv('edges.csv')
edge_index = torch.tensor(edges.values.T, dtype=torch.long)  # 用以匹配 (2, num_edges) 的格式

# 定義模型參數
in_channels = X.size(1)  # 根據節點特徵的維度
hidden_channels = 32
out_channels = 32
num_layers = 3
dropout = 0.5

# 建立模型
model = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)

# 加載權重
model.load_state_dict(model_weights)

# 單一節點特徵
single_node_feature = X[0].unsqueeze(0)  # shape: (1, num_features)

# 對應的邊
single_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# 模型推理
model.eval()
with torch.no_grad():
    # output = model(X, edge_index) # 所有 node
    output = model(single_node_feature, single_edge_index) # 單一 node

# print(output)

predicted_labels = torch.argmax(output, dim=1)
# print(predicted_labels)

result = predicted_labels.item()
print(drink_dict[str(result)])