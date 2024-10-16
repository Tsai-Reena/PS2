#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pickle
import numpy as np
import pandas as pd
import json
import torch

#Then, to reload:
with open('reviewed_edge_index.json', 'r', encoding='utf-8') as file:
    reviewed_edge_index = torch.tensor(json.load(file))

with open('purchased_edge_index.json', 'r', encoding='utf-8') as file:
    purchased_edge_index = torch.tensor(json.load(file))
with open('has_drink_edge_index.json', 'r', encoding='utf-8') as file:
    has_drink_edge_index = torch.tensor(json.load(file))


shop_offset = 0
consumer_offset = 19
drink_offset = 19 + 55
# Read dictionary pkl file


with open('reviewer_to_consumer_id.pkl', 'rb') as fp:
    reviewer_to_consumer_id = pickle.load(fp)
    #print('Person dictionary')
    #print(reviewer_to_consumer_id)
with open('shop_to_shop_id.pkl', 'rb') as fp:
    shop_to_shop_id = pickle.load(fp)
    #print('Person dictionary')
    #print(shop_to_shop_id)
with open('drink_to_drink_id.pkl', 'rb') as fp:
    drink_to_drink_id = pickle.load(fp)
    #print('Person dictionary')
    #print(drink_to_drink_id)



#print(reviewed_edge_index)





user_name = input('輸入要推薦人之名稱:')
target = reviewer_to_consumer_id[user_name]


target_consumer_id = target 
mask = reviewed_edge_index[0] == target_consumer_id
filtered_reviewed_edge_index = reviewed_edge_index[:, mask]

filtered_purchased_edge_index = purchased_edge_index[:, mask]


target_shop_id = filtered_reviewed_edge_index[1][0]  
mask = has_drink_edge_index[0] == target_shop_id
filtered_has_drink_edge_index = has_drink_edge_index[:, mask]
        
# 顯示過濾後的 reviewed_edge_index
#print("Filtered reviewed_edge_index where consumer_id == 0:")
#print(filtered_reviewed_edge_index)
#print(filtered_purchased_edge_index)
#print(filtered_has_drink_edge_index)



# 生成随机的边信息

edges = filtered_reviewed_edge_index.numpy().T
edges_reverse = np.array([[edge[1], edge[0]] for edge in edges])
#print(edges)
#print(edges_reverse)
undirected_edges = np.vstack((edges,edges_reverse))

# 将边信息保存为 CSV 文件
edges_df = pd.DataFrame(undirected_edges, columns=['source', 'target'])
edges_df.to_csv('edges.csv', index=False)


# 生成 370 个节点，每个节点有 99 个特征

num_features = len(reviewed_edge_index[0])-305



feature_filter = reviewed_edge_index[0][0:99].numpy() == target
diagonal = np.eye(len(reviewed_edge_index[0]),dtype = int)
diagonal = np.eye(99,dtype = int)

node_features = np.eye(len(reviewed_edge_index[0])-305,dtype = int)[feature_filter]
#print(np.shape(node_features))

# 将节点特征保存为 CSV 文件
node_features_df = pd.DataFrame(node_features[:][:98], columns=[f'feature{i+1}' for i in range(num_features)])
node_features_df.to_csv('node_features.csv', index=False)


# In[ ]:




