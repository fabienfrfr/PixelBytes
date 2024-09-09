#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

class TokenizedDataset(Dataset):
    def __init__(self, data, tokenizer, seq_length=1024, stride=512):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        self._calculate_total_sequences()

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        item_idx, start_idx = self._get_item_and_start_indices(idx)
        item = self.data[item_idx]
        tokenized = self.tokenizer(text=item.get('text'),
                                   image=item.get('image'),
                                   audio=item.get('audio'))
        input_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
        labels = torch.tensor(tokenized['labels'], dtype=torch.long)
        end_idx = start_idx + self.seq_length
        return {'input_ids': input_ids[start_idx:end_idx],
                'labels': labels[start_idx:end_idx]}

    def _calculate_total_sequences(self):
        self.total_sequences = 0
        self.cumulative_sequences = [0]
        for item in self.data:
            tokenized = self.tokenizer(text=item.get('text'),
                                       image=item.get('image'),
                                       audio=item.get('audio'))
            seq_length = len(tokenized['input_ids'])
            num_sequences = max(1, (seq_length - self.seq_length) // self.stride + 1)
            self.total_sequences += num_sequences
            self.cumulative_sequences.append(self.total_sequences)

    def _get_item_and_start_indices(self, idx):
        item_idx = next(i for i, cum_seq in enumerate(self.cumulative_sequences) if cum_seq > idx) - 1
        relative_idx = idx - self.cumulative_sequences[item_idx]
        start_idx = relative_idx * self.stride
        return item_idx, start_idx

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
                current_input = torch.cat([current_input, next_token], dim=1) # it's wrong : need to call tokenizer
            return current_input

    def train_model(self, dataloader, optimizer, criterion, device, scaler, epochs, accumulation_steps=4):
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            optimizer.zero_grad()
            for i, batch in enumerate(tqdm(dataloader)):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    outputs = self(input_ids)
                    # Mean of all output loss (need)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1)) / accumulation_steps
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * accumulation_steps
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    from tokenizer import ActionPixelBytesTokenizer
    from datasets import load_dataset
    
    def count_parameters_in_k(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000

    tokenizer = ActionPixelBytesTokenizer()
    hf_dataset = load_dataset("ffurfaro/PixelBytes-PokemonAll")
    
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
    dataset = TokenizedDataset(hf_dataset['train'], tokenizer, SEQ_LENGTH, STRIDE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Entraînement
    model.train_model(dataloader, optimizer, criterion, DEVICE, scaler, EPOCHS, ACCUMULATION_STEPS)

    # Sauvegarde du modèle
    model.save_pretrained('lstm_pokemon_sprite_model')

    # Test de génération
    test_input = next(iter(dataloader))['input_ids'][:1].to(DEVICE)
    generated = model.generate(test_input, max_length=100)
    print("Generated sequence:", generated)
