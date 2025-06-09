import pickle
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import ast
from torch.utils.data import TensorDataset
import torch.nn.functional as F 

def get_m3e_embeddings(sentence, model, device):
    if not isinstance(sentence, str) or pd.isna(sentence):
        return torch.zeros(768) 
    embedding = model.encode(sentence, device=device)
    return torch.tensor(embedding)

def create_herb_vector(herbs, herbname_id_dict, vector_length=633):
    herb_ids = torch.zeros(vector_length, dtype=torch.float32)
    for herb in herbs:
        if herb in herbname_id_dict:
            herb_id = herbname_id_dict[herb] - 1
            herb_ids[herb_id] = 1.0    
    return herb_ids

def create_pts(df, name, model, device):
    data = []
    drug_features_df = pd.read_excel('./drug_features.xlsx')
    save_path = f'./{name}.pt'
    for index, row in df.iterrows():
        functions = row['function_description']
        herbs = row['components']
        herbs = [herb.strip() for herb in herbs.split('，')]
        labels = row['label']
        function = get_m3e_embeddings(functions, model, device)
        label = herbname_id_dict[labels]
        indications = []
        herb_ids = create_herb_vector(herbs, herbname_id_dict)

        for herb in herbs:
            matching_rows = drug_features_df[drug_features_df['chinese_name'] == herb]  
            if not matching_rows.empty:
                indication = matching_rows.iloc[0]['indication']
                if isinstance(indication, str):
                    indications.append(indication)
                else:
                    indications.append(str(indication))  
                    print('Not str', herb)
            else:
                indications.append('') 
                print('Not herb', herb)
        #print('indications', indications)
        herbs_feature = get_m3e_embeddings(indications, model, device)

        data_dict = {'function': function, 'label': label, 'herb_features': herbs_feature, 'herb_ids': herb_ids}
        data.append(data_dict)
    torch.save(data, save_path)
    print(f"Data saved to {save_path}", len(data))
    
df_data = pd.read_csv("./ch.csv")
train_df, test_df = train_test_split(df_data, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(df_data, test_size=0.5, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer("../m3e-base", device=device)
herbname_id_dict = torch.load('./herbname_id_dict.pt')
create_pts(test_df, 'test', model=model, device=device)
create_pts(train_df, 'train', model=model, device=device)
create_pts(val_df, 'val', model=model, device=device)

print('Done!')