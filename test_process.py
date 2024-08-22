import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import LabelEncoder
import os
import shutil
import pickle
import json
import numpy as np

class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Cleaned_Uploaded_Output.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 確保文件存在 raw 文件夾中
        source_path = './Cleaned_Uploaded_Output.csv'
        raw_path = os.path.join(self.raw_dir, 'Cleaned_Uploaded_Output.csv')
        if not os.path.exists(raw_path):
            shutil.copyfile(source_path, raw_path)

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(self.raw_paths[0])
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



        ################################################################################edit by limit
        
        # Write file to save edge index information
        reviewed_edge_index_ls = reviewed_edge_index.tolist() #convert to Numpy array
        with open('reviewed_edge_index.json', 'w', encoding='utf-8') as file:
            json.dump(reviewed_edge_index_ls, file, ensure_ascii=False, indent=4)
            
        purchased_edge_index_ls = purchased_edge_index.tolist() #convert to Numpy array
        with open('purchased_edge_index.json', 'w', encoding='utf-8') as file:
            json.dump(purchased_edge_index_ls, file, ensure_ascii=False, indent=4)

        has_drink_edge_index_ls = has_drink_edge_index.tolist() #convert to Numpy array
        with open('has_drink_edge_index.json', 'w', encoding='utf-8') as file:
            json.dump(has_drink_edge_index_ls, file, ensure_ascii=False, indent=4)

        # Building dictionary for matching
        
        reviewer_to_consumer_id = {reviewer: consumer_id + consumer_offset for consumer_id, reviewer in enumerate(label_encoder_consumer.classes_)}
        shop_to_shop_id = {shop: shop_id for shop_id, shop in enumerate(label_encoder_shop.classes_)}
        drink_to_drink_id = {drink: drink_id + drink_offset for drink_id, drink in enumerate(label_encoder_drink.classes_)}

        # Write file to save dictionary
        with open('reviewer_to_consumer_id.pkl', 'wb') as fp:
            pickle.dump(reviewer_to_consumer_id, fp)
            print('dictionary1 saved successfully to file')
        with open('shop_to_shop_id.pkl', 'wb') as fp:
            pickle.dump(shop_to_shop_id, fp)
            print('dictionary2 saved successfully to file')
        with open('drink_to_drink_id.pkl', 'wb') as fp:
            pickle.dump(drink_to_drink_id, fp)
            print('dictionary3 saved successfully to file')

        print("file output successfully")
        ################################################################################edit by limit
        
        
        # Combine edge indices
        edge_index = torch.cat([reviewed_edge_index, purchased_edge_index, has_drink_edge_index], dim=1)
        
        # Create node features as identity matrix
        x = torch.eye(num_nodes, dtype=torch.float)
        
        # Edge attributes (ratings)
        edge_attr = torch.cat([
            torch.tensor(df[['Rating']].values, dtype=torch.float),
            torch.tensor(df[['Drink_Rating']].values, dtype=torch.float),
            torch.ones(len(df), 1)  # For HAS_DRINK relation
        ], dim=0)

        # Dummy labels
        y = torch.zeros(num_nodes, dtype=torch.long)
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:int(0.8 * num_nodes)] = True
        val_mask[int(0.8 * num_nodes):int(0.9 * num_nodes)] = True
        test_mask[int(0.9 * num_nodes):] = True

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

# 指定根目錄
root_path = './dataset/Drink/Drink/'

# 檢查並創建所需的目錄
os.makedirs(os.path.join(root_path, 'raw'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'processed'), exist_ok=True)

# 初始化數據集
dataset = CustomDataset(root=root_path)
dataset.process()