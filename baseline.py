import torch
import torch.nn as nn
import torch.optim as optim
from model_baseline import VAE
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F 
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import math  

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 

set_random_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_function_size = 768
input_herb_features_size = 768
num_classes = 633
batch_size = 64
num_epochs = 1
hidden_size = 1536
latent_size = 256

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
    data = torch.load(f'./{name}.pt')
    return data

train_dataset = load_dataset('train')
test_dataset = load_dataset('test')
val_dataset = load_dataset('val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=list_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=list_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=list_collate)

model = VAE(input_size=input_function_size + input_herb_features_size, 
            hidden_size=hidden_size, latent_size=latent_size, 
            num_classes=num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

def train_vae(model, train_loader, num_epochs=50, lr=0.001, device='cuda'):
    model.train()
    best_precision = 0
    best_epoch = -1
    best_acc = 0
    best_recall = 0
    best_f1 = 0 
    best_perplexity = 1000
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_preds = []
        all_labels = []

        for function, herb_features, labels, lengths, herb_id_batch in train_loader:
            function, herb_features, labels, lengths, herb_id_batch = function.to(device), herb_features.to(device), labels.to(device), lengths.to(device), herb_id_batch.to(device)
            labels = labels-1
            optimizer.zero_grad()
            
            recon_logits = model(function, herb_features, herb_id_batch) 
            loss = model.loss_function(recon_logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(recon_logits, 1)  
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_preds / total_preds

        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        perplexity = torch.exp(torch.tensor(epoch_loss))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Perplexity: {perplexity:.2f}")

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for function, herb_features, labels, lengths, herb_id_batch in val_loader:
                function, herb_features, labels, lengths, herb_id_batch = function.to(device), herb_features.to(device), labels.to(device), lengths.to(device), herb_id_batch.to(device)
                labels = labels-1
                input_seq = torch.cat([function, herb_features], dim=-1)
                recon_logits = model(function, herb_features, herb_id_batch) 
                loss = model.loss_function(recon_logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(recon_logits, 1)  
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct_preds / total_preds

        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        val_perplexity = torch.exp(torch.tensor(val_loss))
        if precision > best_precision:
            torch.save(model.state_dict(), './base.model')
            best_epoch = epoch + 1
            best_precision = precision
            best_acc = val_acc
            best_recall = recall
            best_f1 = f1 
            best_perplexity = val_perplexity
            print(f'Validation Loss:{val_loss:.4f}', 'precision improved at epoch ', best_epoch, '; best_precision:', best_precision, '; best_acc:', best_acc, '; best_recall:', best_recall, '; best_f1:', best_f1, '; best_perplexity:', best_perplexity)
        else:
            print(f'Validation Loss:{val_loss:.4f}', 'No improvement since epoch ', best_epoch, '; best_precision:', best_precision, '; best_acc:', best_acc, '; best_recall:', best_recall, '; best_f1:', best_f1, '; best_perplexity:', best_perplexity)    

nums = 1
name = 'baseline'
for num in range(nums):
    train_vae(model, train_loader, num_epochs=num_epochs, lr=0.001, device=device)
    model.load_state_dict(torch.load('./base.model'))
    model.to(device)
    test_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    for function, herb_features, labels, lengths, herb_id_batch in test_loader:
        function, herb_features, labels, lengths, herb_id_batch = function.to(device), herb_features.to(device), labels.to(device), lengths.to(device), herb_id_batch.to(device)
        labels = labels-1
        input_seq = torch.cat([function, herb_features], dim=-1)
        recon_logits = model(function, herb_features, herb_id_batch) 
        loss = model.loss_function(recon_logits, labels)
        test_loss += loss.item()
        _, predicted = torch.max(recon_logits, 1)  
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
    
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
            
    test_loss /= len(test_loader)
    acc = 100 * correct_preds / total_preds
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    perplexity = torch.exp(torch.tensor(test_loss))
    print(f'Test Loss:{test_loss:.4f}', '; precision:', precision, '; acc:', acc, '; recall:', recall, '; f1:', f1, '; perplexity:', perplexity)
    with open(f"./{name}.txt", "a") as file:  
        file.write(f'{num}, precision: {precision}, acc: {acc}, recall: {recall}, f1: {f1}\n')