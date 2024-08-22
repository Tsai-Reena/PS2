import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import os
import json

file_path = 'Cleaned_Uploaded_Output.csv'
df = pd.read_csv(file_path)

dic = {}

# 建立編碼器
shop_encoder = {name: idx for idx, name in enumerate(df['Shop'].unique())}
consumer_encoder = {name: idx for idx, name in enumerate(df['Reviewer'].unique())}
drink_encoder = {name: idx for idx, name in enumerate(df['Drink'].unique())}

# 添加節點ID
df['shop_id'] = df['Shop'].map(shop_encoder)
df['consumer_id'] = df['Reviewer'].map(consumer_encoder)
df['drink_id'] = df['Drink'].map(drink_encoder)

num_shops = len(shop_encoder)
num_consumers = len(consumer_encoder)
num_drinks = len(drink_encoder)
num_nodes = num_shops + num_consumers + num_drinks

# 建立節點特徵矩陣（單位矩陣）
x = sp.eye(num_nodes, format='csr')
allx = x.copy()

# 根據不同的飲料類型來設定標籤
drink_labels = {name: idx for idx, name in enumerate(df['Drink'].unique())}

# 為每個飲料節點設計標籤
y = np.zeros((num_nodes, 1))
for _, row in df.iterrows():
    drink_id = row['drink_id'] + num_shops + num_consumers
    y[drink_id] = drink_labels[row['Drink']]

ally = y.copy()

# 建立圖結構
graph = {}
for _, row in df.iterrows():
    shop_id = row['shop_id']
    consumer_id = row['consumer_id'] + num_shops
    drink_id = row['drink_id'] + num_shops + num_consumers
    rating = row['Rating']
    
    # REVIEWED 關係
    if shop_id not in graph:
        graph[shop_id] = []
    if consumer_id not in graph:
        graph[consumer_id] = []
    graph[shop_id].append(consumer_id)
    graph[consumer_id].append(shop_id)
    
    # PURCHASED 關係
    if consumer_id not in graph:
        graph[consumer_id] = []
    if drink_id not in graph:
        graph[drink_id] = []
    graph[consumer_id].append(drink_id)
    graph[drink_id].append(consumer_id)
    
    # HAS_DRINK 關係
    if shop_id not in graph:
        graph[shop_id] = []
    if drink_id not in graph:
        graph[drink_id] = []
    graph[shop_id].append(drink_id)
    graph[drink_id].append(shop_id)

# 劃分訓練、驗證和測試節點
train_indices = np.arange(0, int(0.7 * num_nodes))
test_indices = np.arange(int(0.7 * num_nodes), int(0.85 * num_nodes))
val_indices = np.arange(int(0.85 * num_nodes), num_nodes)

# 構建特徵矩陣
tx = x[test_indices].toarray()
ty = y[test_indices]
x = x[train_indices].toarray()
allx = allx[np.concatenate((train_indices, val_indices))].toarray()
ally = ally[np.concatenate((train_indices, val_indices))]

# 指定保存路徑
save_path = './dataset/Drink/Drink/raw'
os.makedirs(save_path, exist_ok=True)

# 保存文件
with open(os.path.join(save_path, 'ind.custom.x'), 'wb') as f:
    pickle.dump(x, f)

with open(os.path.join(save_path, 'ind.custom.allx'), 'wb') as f:
    pickle.dump(allx, f)

with open(os.path.join(save_path, 'ind.custom.y'), 'wb') as f:
    pickle.dump(y, f)

with open(os.path.join(save_path, 'ind.custom.ally'), 'wb') as f:
    pickle.dump(ally, f)

with open(os.path.join(save_path, 'ind.custom.tx'), 'wb') as f:
    pickle.dump(tx, f)

with open(os.path.join(save_path, 'ind.custom.ty'), 'wb') as f:
    pickle.dump(ty, f)

with open(os.path.join(save_path, 'ind.custom.graph'), 'wb') as f:
    pickle.dump(graph, f)

with open(os.path.join(save_path, 'ind.custom.test.index'), 'wb') as f:
    np.savetxt(f, test_indices, fmt='%d')

for drink, label in drink_labels.items():
    dic[str(label)] = str(drink)
    print(f"Drink: {drink}, Label: {label}")

# print(dic)

with open('labels.json', 'w', encoding='utf-8') as fp:
    json.dump(dic, fp, ensure_ascii=False)