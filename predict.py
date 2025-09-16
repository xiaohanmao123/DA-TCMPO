import torch
import torch.nn as nn
import torch.optim as optim
from model import VAE
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F 
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import math  
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import ast
from torch.utils.data import TensorDataset 
import re

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  

set_random_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_function_size = 768
input_herb_features_size = 768
num_classes = 633
batch_size = 64
num_epochs = 300
hidden_size = 1536
latent_size = 256

model = SentenceTransformer("./m3e-base", device=device)
herbname_id_dict = torch.load('./herbname_id_dict.pt')
id_herbname_dict = torch.load('./herb_id_dict.pt')

def list_collate(batch):
    function_batch = [item['function'] for item in batch]
    label_batch = [item['label'] for item in batch]
    herb_features = [item['herb_features'] for item in batch]
    herb_ids = [item['herb_ids'] for item in batch]
    function_batch = torch.stack(function_batch) 
    herb_id_batch = torch.stack(herb_ids) 
    label_batch = torch.tensor(label_batch).long()  
    herb_features_batch = pad_sequence(herb_features, batch_first=True, padding_value=0)
    if herb_features_batch.shape[1] < input_herb_features_size:
        padding_size = input_herb_features_size - herb_features_batch.shape[1]
        herb_features_batch = F.pad(herb_features_batch, (0, padding_size), value=0)
    else:
        herb_features_batch = herb_features_batch[:, :input_herb_features_size]
    seq_lengths = torch.tensor([len(seq) for seq in herb_features]) 
    return function_batch, herb_features_batch.float(), label_batch, seq_lengths, herb_id_batch

def load_dataset(name):
    data = torch.load(f'{name}.pt')
    return data

predict_dataset = load_dataset('predict')
predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, collate_fn=list_collate)

model = VAE(input_size=input_function_size + input_herb_features_size, 
            hidden_size=hidden_size, latent_size=latent_size, 
            num_classes=num_classes).to(device)
model.load_state_dict(torch.load('./mo.model'))
with torch.no_grad():
    all_preds = []
    all_labels = []
    indices = []
    for function, herb_features, labels, lengths, herb_id_batch in predict_loader:
        function, herb_features, labels, lengths, herb_id_batch = function.to(device), herb_features.to(device), labels.to(device), lengths.to(device), herb_id_batch.to(device)
        labels = labels-1
        input_seq = torch.cat([function, herb_features], dim=-1)
        recon_logits, mean, logvar = model(function, herb_features, herb_id_batch)
        _, predicted = torch.max(recon_logits, 1) 
        topk_values, topk_indices = torch.topk(recon_logits, 10, dim=1)
        indices.extend(topk_indices.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    df_logits = pd.DataFrame({"Logits": indices})

def trans_name(all_preds):
    lists = []
    for pred in all_preds:
        pred_name = id_herbname_dict[pred]
        lists.append(pred_name)
    return lists

pred_list = trans_name(all_preds)
label_list = trans_name(all_labels)

def convert_logits_to_names(logits_str):
    numbers = list(map(int, re.findall(r"\d+", str(logits_str))))
    herb_names = [id_herbname_dict[num] for num in numbers if num in id_herbname_dict]
    return ", ".join(herb_names)

df_logits["HerbNames"] = df_logits["Logits"].apply(convert_logits_to_names)
df_logits["PredNames"] = pred_list
df_logits["LabelNames"] = label_list
df_logits = df_logits.drop(columns=["Logits"])
df_logits.to_csv("./predictions2.csv", index=False, encoding="utf-8")