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
import numpy as np, pandas as pd
from transformers import PreTrainedModel, PretrainedConfig
from mambapy.mamba import Mamba, MambaConfig
from torch.cuda.amp import GradScaler

## Data Part
class TokenPxByDataset(Dataset):
    def __init__(self, data, tokenizer, seq_length=1024, stride=512):
        self.seq_length = seq_length
        self.stride = stride
        self.tokenized_data = []
        self._preprocess_data(data, tokenizer)

    def _preprocess_data(self, data, tokenizer):
        self.total_sequences = 0
        for item in data:
            tokenized = tokenizer(text=item.get('text'),
                                  image=item.get('image'),
                                  audio=item.get('audio'))
            length = len(tokenized['input_ids'])
            num_sequences = max(1, (length - self.seq_length) // self.stride + 1)
            self.tokenized_data.append((tokenized, length, num_sequences))
            self.total_sequences += num_sequences

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        idx = idx % self.total_sequences
        item_idx, start_idx = self._get_item_and_start_indices(idx)
        tokenized, length, _ = self.tokenized_data[item_idx]
        end_idx = start_idx + self.seq_length

        input_ids = tokenized['input_ids'][start_idx:end_idx % length]
        labels = tokenized['labels'][start_idx:end_idx % length]
        return {'input_ids': torch.as_tensor(input_ids, dtype=torch.long), 
                'labels': torch.as_tensor(labels, dtype=torch.long)}

    def _get_item_and_start_indices(self, idx):
        for i, (_, _, num_sequences) in enumerate(self.tokenized_data):
            if idx < num_sequences:
                start_idx = (idx * self.stride) % self.tokenized_data[i][1]
                return i, start_idx
            idx -= num_sequences
        raise IndexError("Index out of range")

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {'input_ids': input_ids_padded,'labels': labels_padded}

## Model and training
class ModelConfig(PretrainedConfig):
    def __init__(self, vocab_size=2048, embed_size=256, hidden_size=512, num_layers=2, pxby_dim=6, 
                 auto_regressive=False, model_type="lstm", d_conv=4,expand=2,**kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.pxby_dim = pxby_dim
        self.pxby_emb = embed_size // (self.pxby_dim * (expand if model_type=="mamba" else 1))
        self.embed_size = int(self.pxby_emb * self.pxby_dim)
        self.AR = auto_regressive
        self.model_type = model_type
        self.hidden_size = self.d_state = hidden_size
        # Mamba specific attributes
        self.d_conv = d_conv
        self.expand = expand
    def get_mamba_config(self):
        return MambaConfig(d_model=self.embed_size, n_layers=self.num_layers, 
                           d_state=self.d_state, d_conv=self.d_conv, expand_factor=self.expand)

class aPxBySequenceModel(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = "hybrid"

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.pxby_emb, padding_idx=0)
        if config.model_type == "mamba": self.sequence_model = Mamba(config.get_mamba_config())
        else: self.sequence_model = nn.LSTM(config.embed_size, config.hidden_size, config.num_layers, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size * (config.pxby_dim if config.AR else 1))
        self.pxby_dim, self.AR, self.model_type = config.pxby_dim, config.AR, config.model_type

    def forward(self, input_ids):
        batch_size, seq_len, _ = input_ids.shape
        embedded = self.embedding(input_ids).view(batch_size, seq_len, -1)
        if self.model_type == "mamba": output = self.sequence_model(embedded)
        else: output, _ = self.sequence_model(embedded)
        output = self.fc(output) # Shape: (batch_size, seq_len, vocab_size*pxby) or (batch_size, seq_len, vocab_size)
        return output.view(batch_size, seq_len, self.pxby_dim, -1) if self.AR else output

    def generate(self, input_ids, num_generate, temperature=1.0):
        self.eval()
        with torch.no_grad():
            current_input = input_ids.clone()
            for i in range(num_generate): # Generate next token
                outputs = self(current_input)
                next_token = torch.multinomial(
                    torch.softmax(outputs[:, -1, -1] if self.AR else outputs[:, -1] / temperature, dim=-1),
                    num_samples=1)
                if self.AR: current_input = torch.cat([current_input, next_token.unsqueeze(1).unsqueeze(1)], dim=2) # true generator
                else: current_input[:, -(i+1)] = next_token.squeeze(-1) # Replace the (i+1)th token from the end (False generator)
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
                    self.save_model(is_best=True)
        self.save_model(is_best=False)
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
                    if self.AR: # Reshape and shift for autoregressive mode
                        outputs = outputs[:, :-1].contiguous().view(-1, self.config.vocab_size)
                        target = input_ids[:, 1:].contiguous().view(-1)
                    else: # Flatten outputs and use labels as target for non-autoregressive mode
                        outputs = outputs.view(-1, self.config.vocab_size)
                        target = labels.view(-1)
                    loss = criterion(outputs, target)
                    if is_training:
                        loss = loss / accumulation_steps
                if is_training:
                    (scaler.scale(loss) if scaler else loss).backward()
                    if (i + 1) % accumulation_steps == 0:
                        (scaler.step(optimizer) if scaler else optimizer.step())
                        (scaler.update() if scaler else None)
                        optimizer.zero_grad()
                total_loss += loss.item() * (accumulation_steps if is_training else 1)
                total_correct += (outputs.view(-1, outputs.size(-1)).argmax(-1) == target.view(-1)).sum().item()
                total_samples += target.numel()
        return total_loss / len(dataloader), total_correct / total_samples

    def save_model(self, is_best=False):
        save_dir = os.path.join(os.getcwd(), f"{self.model_type}_{'autoregressive' if self.AR else 'predictive'}_{('best' if is_best else 'last')}")
        os.makedirs(save_dir, exist_ok=True)
        self.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

if __name__ == '__main__':
    from tokenizer import ActionPixelBytesTokenizer
    from datasets import load_dataset
    
    def count_parameters_in_k(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000

    hf_dataset = load_dataset("ffurfaro/PixelBytes-PokemonAll")['train'].train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = hf_dataset['train'], hf_dataset['test']
    
    DATA_REDUCTION = 6
    tokenizer = ActionPixelBytesTokenizer(data_slicing=DATA_REDUCTION)
    
    # Paramètres
    VOCAB_SIZE = tokenizer.vocab_size
    EMBED_SIZE = 128
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    PXBY_DIM = 6 # tokenizer
    AR = False
    MODEL_TYPE = "lstm"
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ACCUMULATION_STEPS = 4
    SEQ_LENGTH = 1024
    STRIDE = 512
    
    # Initialisation du modèle
    config = ModelConfig(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, 
                          num_layers=NUM_LAYERS, pxby_dim=PXBY_DIM, auto_regressive=AR, model_type=MODEL_TYPE)
    model = aPxBySequenceModel(config).to(DEVICE)
    print(f"Le modèle a {count_parameters_in_k(model):.2f}k paramètres entraînables.")

    # Préparation des données
    def dataloading(ds):
        dataset = TokenPxByDataset(ds, tokenizer, SEQ_LENGTH, STRIDE)
        return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
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
