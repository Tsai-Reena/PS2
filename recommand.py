import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle

from model import AutoLink_l3, SearchGraph_l31  # 確保正確導入模型類


def load_custom_data(file_path):
    df = pd.read_csv(file_path)

    label_encoder_shop = LabelEncoder()
    label_encoder_consumer = LabelEncoder()
    label_encoder_drink = LabelEncoder()
    
    df['shop_id'] = label_encoder_shop.fit_transform(df['Shop'])
    df['consumer_id'] = label_encoder_consumer.fit_transform(df['Reviewer'])
    df['drink_id'] = label_encoder_drink.fit_transform(df['Drink'])

    num_shops = len(label_encoder_shop.classes_)
    num_consumers = len(label_encoder_consumer.classes_)
    num_drinks = len(label_encoder_drink.classes_)
    num_nodes = num_shops + num_consumers + num_drinks
    
    shop_offset = 0
    consumer_offset = num_shops
    drink_offset = num_shops + num_consumers

    # Create edge indices for different relations
    reviewed_edge_index = torch.tensor([df['consumer_id'].values + consumer_offset, df['shop_id'].values + shop_offset], dtype=torch.long)
    purchased_edge_index = torch.tensor([df['consumer_id'].values + consumer_offset, df['drink_id'].values + drink_offset], dtype=torch.long)
    has_drink_edge_index = torch.tensor([df['shop_id'].values + shop_offset, df['drink_id'].values + drink_offset], dtype=torch.long)
    
    # Combine edge indices
    edge_index = torch.cat([reviewed_edge_index, purchased_edge_index, has_drink_edge_index], dim=1)
    
    # Create node features as identity matrix
    x = torch.eye(num_nodes, dtype=torch.float)
    
    # Edge attributes (ratings)
    edge_attr = torch.cat([
        torch.tensor(df[['Rating']].values, dtype=torch.float),
        torch.tensor(df[['Drink_Rating']].values, dtype=torch.float),
        torch.ones((has_drink_edge_index.size(1), 1))  # For HAS_DRINK relation
    ], dim=0)

    # Dummy labels
    y = torch.zeros(num_nodes, dtype=torch.long)
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Adjust split according to number of nodes
    train_mask[:int(0.8 * num_consumers)] = True
    val_mask[int(0.8 * num_consumers):int(0.9 * num_consumers)] = True
    test_mask[int(0.9 * num_consumers):num_consumers] = True

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data, label_encoder_shop, label_encoder_consumer, label_encoder_drink

def load_encoders_and_model(shop_encoder_path, consumer_encoder_path, drink_encoder_path, model_path, arch_net_path, num_features, hidden_channels, num_layers, dropout, use_sage, lin_layers, cat_type, arch_dim, arch_layers, temperature):
    with open(shop_encoder_path, 'rb') as f:
        shop_encoder = pickle.load(f)

    with open(consumer_encoder_path, 'rb') as f:
        consumer_encoder = pickle.load(f)

    with open(drink_encoder_path, 'rb') as f:
        drink_encoder = pickle.load(f)

    # 初始化模型
    model = AutoLink_l3(num_features, hidden_channels, num_layers, dropout, use_sage, lin_layers=lin_layers, cat_type=cat_type)
    arch_net = SearchGraph_l31(hidden_channels, arch_dim, arch_layers, cat_type=cat_type, temperature=temperature)
    
    # 加載預訓練模型權重
    model.load_state_dict(torch.load(model_path))
    arch_net.load_state_dict(torch.load(arch_net_path))  
    model.eval()
    arch_net.eval()
    
    return shop_encoder, consumer_encoder, drink_encoder, model, arch_net

# 定義推薦函數
def recommend_shops(consumer_name, shop_encoder, consumer_encoder, drink_encoder, model, arch_net, data):
    # 轉換消費者名稱為ID
    consumer_id = consumer_encoder.transform([consumer_name])[0] + len(shop_encoder.classes_)
    
    # 構建輸入圖
    num_shops = len(shop_encoder.classes_)
    num_consumers = len(consumer_encoder.classes_)
    num_drinks = len(drink_encoder.classes_)
    num_nodes = num_shops + num_consumers + num_drinks
    
    # 單位矩陣作為特徵矩陣
    x = torch.eye(num_nodes, dtype=torch.float).to(data.x.device)
    
    # 構建推薦的邊
    consumer_indices = torch.tensor([consumer_id] * num_shops, dtype=torch.long)
    shop_indices = torch.tensor(list(range(num_shops)), dtype=torch.long)
    pos_edge = torch.stack([consumer_indices, shop_indices], dim=0).to(data.x.device)
    
    # 模型預測
    with torch.no_grad():
        h = model(x, data.edge_index)
        preds = model.compute_pred(h, arch_net, pos_edge).squeeze().cpu()
    
    # 提取得分最高的商店ID
    predicted_shop_id = preds.argmax().item()
    
    # 反向編碼
    shop_name = shop_encoder.inverse_transform([predicted_shop_id])[0]
    
    return shop_name

# 使用示例
if __name__ == "__main__":
    # 文件路徑
    shop_encoder_path = 'shop_encoder.pkl'
    consumer_encoder_path = 'consumer_encoder.pkl'
    drink_encoder_path = 'drink_encoder.pkl'
    model_path = 'weight/l3bl-GCN_GCN_Custom_arch-3-64-0.1_3-lin3_lr0.001-tem0.07_hidd32-32_multi_model_0.pth'
    arch_net_path = 'weight/l3bl-GCN_GCN_Custom_arch-3-64-0.1_3-lin3_lr0.001-tem0.07_hidd32-32_multi_arch_0.pth'
    data_path = 'Cleaned_Uploaded_Output.csv'
    
    # 加載數據
    data, shop_encoder, consumer_encoder, drink_encoder = load_custom_data(data_path)
    num_features = data.num_node_features
    data.edge_index = to_undirected(data.edge_index)
    device = torch.device('cpu')
    data = data.to(device)
    
    # 模型參數
    hidden_channels = 32
    num_layers = 3
    dropout = 0.1
    use_sage = 'GCN'
    lin_layers = 3
    cat_type = 'multi'
    arch_dim = 64
    arch_layers = 3
    temperature = 0.07

    # 加載編碼器和模型
    shop_encoder, consumer_encoder, drink_encoder, model, arch_net = load_encoders_and_model(
        shop_encoder_path, consumer_encoder_path, drink_encoder_path, model_path, arch_net_path,
        num_features, hidden_channels, num_layers, dropout, use_sage, lin_layers, cat_type, arch_dim, arch_layers, temperature)
    
    # 提示用戶輸入消費者名稱
    consumer_name = input("請輸入消費者名稱：")

    # 進行推薦
    recommended_shop = recommend_shops(consumer_name, shop_encoder, consumer_encoder, drink_encoder, model, arch_net, data)
    print(f'Recommended shop for {consumer_name}: {recommended_shop}')
