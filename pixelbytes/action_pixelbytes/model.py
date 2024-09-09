#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import os
import torch, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

class TokenizedDataset(Dataset):
    def __init__(self, data, tokenizer, seq_length=1024, stride=512):
        self.seq_length = seq_length
        self.stride = stride
        self.tokenized_data = []
        self._preprocess_data(data, tokenizer)

    def _preprocess_data(self, data, tokenizer):
        for item in data :
            tokenized = tokenizer(text=item.get('text'),
                                  image=item.get('image'),
                                  audio=item.get('audio'))
            self.tokenized_data.append(tokenized)
        self._calculate_total_sequences()

    def _calculate_total_sequences(self):
        self.total_sequences = sum(max(1, (len(item['input_ids']) - self.seq_length) // self.stride + 1)
                                   for item in self.tokenized_data)
    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        item_idx, start_idx = self._get_item_and_start_indices(idx)
        tokenized = self.tokenized_data[item_idx]
        end_idx = start_idx + self.seq_length
        return {'input_ids': torch.as_tensor(tokenized['input_ids'][start_idx:end_idx], dtype=torch.long),
                'labels': torch.as_tensor(tokenized['labels'][start_idx:end_idx], dtype=torch.long)}

    def _get_item_and_start_indices(self, idx):
        for i, item in enumerate(self.tokenized_data):
            num_sequences = max(1, (len(item['input_ids']) - self.seq_length) // self.stride + 1)
            if idx < num_sequences:
                return i, idx * self.stride
            idx -= num_sequences
        raise IndexError("Index out of range")
    
    def get_num_sequences(self, item):
        return max(1, (len(item['input_ids']) - self.seq_length) // self.stride + 1)

class ShuffledSampler(Sampler):
    def __init__(self, data_source, seed=None):
        self.data_source = data_source
        self.seed = seed
        self.indices = list(range(len(data_source)))
        
    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(self.indices)
        for idx in self.indices:
            yield idx

    def __len__(self):
        return len(self.data_source)

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {'input_ids': input_ids_padded,'labels': labels_padded}

class ModelConfig_(PretrainedConfig):
    model_type = "lstm"
    def __init__(self, vocab_size=2048, embed_size=256, hidden_size=512, num_layers=2, pxby_dim=6, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pxby_dim = pxby_dim

class BestPreTrainedModel(PreTrainedModel):
    config_class = ModelConfig_
    base_model_prefix = "lstm"

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.lstm = nn.LSTM(config.embed_size * config.pxby_dim, config.hidden_size, config.num_layers, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len, context_size = input_ids.shape
        embedded = self.embedding(input_ids).view(batch_size, seq_len, -1)
        output, _ = self.lstm(embedded)
        return self.fc(output)

    def generate(self, input_ids, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            current_input = input_ids
            for _ in range(max_length - input_ids.size(1)):
                outputs = self(current_input)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                current_input = torch.cat([current_input, next_token], dim=1) # it's wrong : need to tokenizer decode
            return current_input

    def train_model(self, train_dataloader, val_dataloader, optimizer, criterion, device, scaler, epochs, accumulation_steps=4, eval_every=5):
        best_loss = float('inf')
        val_loss, val_accuracy = self._process_epoch(val_dataloader, None, criterion, device, None, accumulation_steps)
        metrics = [{'epoch': 0,'train_loss': val_loss,'train_accuracy': val_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy}]
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        for epoch in range(epochs):
            train_loss, train_accuracy = self._process_epoch(train_dataloader, optimizer, criterion, device, scaler, accumulation_steps)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                val_loss, val_accuracy = self._process_epoch(val_dataloader, None, criterion, device, None, accumulation_steps)
                metrics.append({'epoch': epoch + 1,'train_loss': train_loss,'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_pretrained(os.path.join(os.getcwd(), "best_model"))
                    print(f"Best model saved with validation loss: {best_loss:.4f}")
        self.save_pretrained(os.path.join(os.getcwd(), "last_model"))
        df_metrics = pd.DataFrame(metrics); df_metrics.to_csv('training_metrics.csv', index=False)
        
    def _process_epoch(self, dataloader, optimizer, criterion, device, scaler, accumulation_steps):
        is_training = optimizer is not None
        self.train(is_training)
        total_loss = total_correct = total_samples = 0
        with torch.set_grad_enabled(is_training):
            for i, batch in enumerate(tqdm(dataloader, desc="Training" if is_training else "Evaluating")):
                input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
                with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                    outputs = self(input_ids)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    if is_training:
                        loss = loss / accumulation_steps
                if is_training:
                    (scaler.scale(loss) if scaler else loss).backward()
                    if (i + 1) % accumulation_steps == 0:
                        (scaler.step(optimizer) if scaler else optimizer.step())
                        (scaler.update() if scaler else None)
                        optimizer.zero_grad()
                total_loss += loss.item() * (accumulation_steps if is_training else 1)
                total_correct += (outputs.view(-1, outputs.size(-1)).argmax(-1) == labels.view(-1)).sum().item()
                total_samples += labels.numel()
        return total_loss / len(dataloader), total_correct / total_samples

if __name__ == '__main__':
    from tokenizer import ActionPixelBytesTokenizer
    from datasets import load_dataset
    
    def count_parameters_in_k(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000

    hf_dataset = load_dataset("ffurfaro/PixelBytes-PokemonAll")['train'].train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = hf_dataset['train'], hf_dataset['test']
    
    DATA_REDUCTION = 4
    tokenizer = ActionPixelBytesTokenizer(data_slicing=DATA_REDUCTION)
    
    # Paramètres
    VOCAB_SIZE = tokenizer.vocab_size
    EMBED_SIZE = 128
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    PXBY_DIM = 6 # tokenizer
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ACCUMULATION_STEPS = 4
    SEQ_LENGTH = 1024
    STRIDE = 512
    
    # Initialisation du modèle
    config = ModelConfig_(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, pxby_dim=PXBY_DIM)
    model = BestPreTrainedModel(config).to(DEVICE)
    print(f"Le modèle a {count_parameters_in_k(model):.2f}k paramètres entraînables.")

    # Préparation des données
    def dataloading(ds):
        dataset = TokenizedDataset(ds, tokenizer, SEQ_LENGTH, STRIDE)
        sampler = ShuffledSampler(dataset, seed=42)
        return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, sampler=sampler)
    train_dataloader, val_dataloader = dataloading(train_ds), dataloading(val_ds)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Entraînement
    model.train_model(train_dataloader, val_dataloader, optimizer, criterion, DEVICE, scaler, EPOCHS, ACCUMULATION_STEPS)

    # Sauvegarde du modèle
    model.save_pretrained('lstm_pokemon_sprite_model')

    # Test de génération
    test_input = next(iter(dataloader))['input_ids'][:1].to(DEVICE)
    generated = model.generate(test_input, max_length=100)
    print("Generated sequence:", generated)
