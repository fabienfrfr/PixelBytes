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
                 bidirection=False, objective=1, num_diffusion_steps=5, model_type="lstm", custom_model=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.pxby_dim = pxby_dim
        self.pxby_emb = embed_size // self.pxby_dim
        self.bidirection = bidirection
        self.embed_size = int(self.pxby_emb * self.pxby_dim) 
        self.objective = ["predict","autoregressive","diffusion"][objective] if isinstance(objective, int) else objective # crafts
        self.num_diffusion_steps = num_diffusion_steps # + 1 in fact
        self.model_type = model_type # Don't forget mask if you want to use transformer
        self.custom_model = custom_model
        self.hidden_size = hidden_size // (1 + bidirection) if isinstance(objective, int) else hidden_size # crafts

class aPxBySequenceModel(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = "hybrid"

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.pxby_emb, padding_idx=0)
        if config.model_type != "lstm": self.sequence_model = config.custom_model(**config)
        else: self.sequence_model = nn.LSTM(config.embed_size, config.hidden_size, config.num_layers, bidirectional=config.bidirection, batch_first=True)
        self.fc = nn.Linear(config.hidden_size * (1 + config.bidirection), config.vocab_size * (1 if config.objective=="predict" else config.pxby_dim))
        self.pxby_dim, self.num_diffusion_steps = config.pxby_dim, config.num_diffusion_steps
        self.objective, self.model_type, self.pxby_emb = config.objective, config.model_type, config.pxby_emb

    def forward(self, x, t=None,m=None):
        batch_size, seq_len, _ = x.shape
        x = self.embedding(x).view(batch_size, seq_len, -1)
        if self.objective == "diffusion": # prefer bidirectionnal
            if t is None: t = torch.randint(self.num_diffusion_steps-1, self.num_diffusion_steps+1, (batch_size,), device=x.device)
            if m is None: 
                p = torch.randint(0, seq_len, (int(3*seq_len/4.),), device=x.device).unsqueeze(-1) # position (with repeat)
                m = torch.ones((batch_size, seq_len, self.pxby_dim, self.pxby_emb), device=x.device); m[:,p] = 0 # complete mask
                m[:,torch.clamp(p + 1, max=seq_len-1), :-1] = 0; m = m.view(batch_size, seq_len, -1) # print(torch.unique(torch.concat([p,torch.clamp(p + 1, max=seq_len-1)])).shape)
            alpha_t, noise = (1 - t / self.num_diffusion_steps)[:, None, None], torch.randn_like(x)
            x = torch.where(m == 1, x, (1 - alpha_t) * noise +  alpha_t * x)
        x, _ = self.sequence_model(x)
        x = self.fc(x) # Shape: (batch_size, seq_len, vocab_size*pxby) or (batch_size, seq_len, vocab_size)
        return x if self.objective=="predict" else x.view(batch_size, seq_len, self.pxby_dim, -1)

    def _process_probs(self, outputs, temperature, current_input, pos=None):
        if self.objective == "predict":
            probs = torch.softmax(outputs[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_input[:, -(pos+1)] = next_token.squeeze(-1) # Replace the (i+1)th token from the end (False generator)
            return current_input
        elif self.objective == "diffusion":
            probs = torch.softmax(outputs[:, pos].view(-1, self.config.vocab_size) / temperature, dim=-1)
            pos_tokens = torch.multinomial(probs, num_samples=1).view(1, len(pos), -1)
            current_input.scatter_(1, pos.expand(1, -1, self.pxby_dim), pos_tokens)
            return current_input    
        else : # Reshape outputs to [1, vocab_size], apply softmax and sample from prob distribution
            probs = torch.softmax(outputs[:, -1].view(-1, self.config.vocab_size) / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).view(1, 1, -1)
            return torch.cat([current_input, next_token], dim=1) # true generator

    def generate(self, input_ids, idn_generator=None, temperature=1.0):
        self.eval()
        with torch.no_grad():
            current_input, device = input_ids.clone(), input_ids.device
            batch_size, seq_len = current_input.shape[:2]
            if self.objective == "diffusion" : # Generate n specific position (setpoint in control problem)
                position = torch.as_tensor(idn_generator, dtype=torch.long, device=device)[:,None] if idn_generator is not None else torch.randint(0, seq_len, (10,), device=device).unsqueeze(-1)
                mask = torch.ones((batch_size, seq_len, self.pxby_dim, self.pxby_emb)); mask[:,position] = 0 # complete mask
                mask[:,torch.clamp(position + 1, max=seq_len-1), :-1] = 0; mask = mask.view(batch_size, seq_len, -1)
                position = torch.unique(torch.concat([position,torch.clamp(position + 1, max=seq_len-1)])).unsqueeze(-1)
                for t in reversed(range(self.num_diffusion_steps+1)):
                    t_tensor = torch.full((batch_size,), t, device=device)
                    outputs = self(current_input, t_tensor, mask)
                    current_input = self._process_probs(outputs, temperature, current_input, position)
            else :
                idn_generator = idn_generator if idn_generator is not None else 10
                for i in range(idn_generator): # Generate next token
                    outputs = self(current_input)
                    current_input = self._process_probs(outputs, temperature, current_input, i)
            return current_input

    def train_model(self, train_dataloader, val_dataloader, epochs, optimizer=None, criterion=None, accumulation_steps=4, eval_every=5):
        if self.objective != "diffusion" and self.bidirection : print("[WARNING] predictive model know the futur for these config..")
        best_loss = float('inf')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) if optimizer is None else optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=-100) if criterion is None else criterion
        scaler = GradScaler() if torch.cuda.is_available() else None
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
                    if self.objective=="autoregressive": # Reshape and shift for autoregressive mode
                        outputs = self(input_ids[:, :-1]).contiguous().view(-1, self.config.vocab_size)
                        target = input_ids[:, 1:].contiguous().view(-1)
                    elif self.objective=="diffusion":
                        outputs = self(input_ids).contiguous().view(-1, self.config.vocab_size)
                        target = input_ids.contiguous().view(-1) 
                    else: # Flatten outputs and use labels as target for non-autoregressive mode
                        outputs = self(input_ids).view(-1, self.config.vocab_size)
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
        save_dir = os.path.join(os.getcwd(), f"{self.model_type}_{self.objective}_{('best' if is_best else 'last')}")
        os.makedirs(save_dir, exist_ok=True)
        self.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

if __name__ == '__main__':
    #from tokenizer import ActionPixelBytesTokenizer
    #from datasets import load_dataset
    ## some test in test.py file (here very basic)
    config = ModelConfig(objective=1, bidirection=True)
    #model = aPxBySequenceModel.from_pretrained("ffurfaro/aPixelBytes-Pokemon", subfolder="lstm_autoregressive_last")
    model = aPxBySequenceModel.from_pretrained("ffurfaro/aPixelBytes-OptimalControl", subfolder="bilstm_diffusion_last")#, ignore_mismatched_sizes=True)
    #model = aPxBySequenceModel(config)
    input_tensor = torch.randint(0, 151, (1, 1024, 6))
    direct_output = model(input_tensor)
    output_tensor = model.generate(input_tensor, idn_generator=[1021,1022]) # inconsistent with noise (diffusion uncomment)
    print(input_tensor, output_tensor, output_tensor.shape, torch.where(input_tensor - output_tensor !=0)) # ok
