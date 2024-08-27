#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import f1_score
from transformers import PreTrainedModel, PretrainedConfig
from dataclasses import dataclass

from .model import *

@dataclass
class TrainConfig:
    hf_token : str
    repo_name : str
    model : PreTrainedModel
    model_config : PretrainedConfig
    train_dataset : DataLoader
    test_dataset : DataLoader
    batch_size : int = 32
    learning_rate : int = 0.001
    num_epochs: int = 10
    save_dir : str = "models"
    dataset_name : str = "dataset"
    eval_every : int = 5

    def __post_init__(self):
        self.info = "_".join([self.model.name, 
                     "bi" if self.model_config.bidirectional else "uni", 
                     "pxby" if self.model_config.pxby_embed else "center",
                     str(self.model_config.dim) + "-dim",
                     str(self.model_config.d_state) + "-state",
                     str(self.model_config.depth) + "-layer"])
class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.config.model.to(self.device)
        self.save_dir = os.path.join(config.save_dir, f"{config.info}_{config.dataset_name}")
        self.hf_dir = os.path.join(config.save_dir, f"{config.info}")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.hf_dir+"_best", exist_ok=True); os.makedirs(self.hf_dir+"_last", exist_ok=True)
        print("Complete path of pytorch model '.pth': " + self.save_dir)
        self.train_loader = DataLoader(config.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(config.test_dataset, batch_size=config.batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.num_epochs = config.num_epochs
        self.eval_every = config.eval_every
        self.results = []
        self.test_size = len(config.test_dataset)
        self.hf_token = config.hf_token
        self.repo_name = config.dataset_name if (not(self.hf_token is None) and config.repo_name is None) else config.repo_name

    def train_and_evaluate(self):
        if self.hf_token is None : print('No HF token given..')
        best_test_loss = float('inf')
        test_metrics = self._evaluate(self.test_loader)
        self.results.append({'epoch': 0, 'train_eval_loss' : test_metrics['loss'], 'train_accuracy': test_metrics['accuracy'], 'train_f1': test_metrics['f1'],
                             'test_loss': test_metrics['loss'],'test_accuracy': test_metrics['accuracy'], 'test_f1': test_metrics['f1']})
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            train_loss = self._train_epoch()
            if (epoch + 1) % self.eval_every == 0 or epoch == self.num_epochs - 1:
                train_metrics = self._evaluate(self.train_loader, max_samples=self.test_size)
                test_metrics = self._evaluate(self.test_loader)
                self.results.append({'epoch': epoch + 1, 'train_eval_loss': train_metrics['loss'],'train_accuracy': train_metrics['accuracy'], 'train_f1': train_metrics['f1'],
                                     'test_loss': test_metrics['loss'],'test_accuracy': test_metrics['accuracy'], 'test_f1': test_metrics['f1']})       
                if test_metrics['loss'] < best_test_loss:
                    best_test_loss = test_metrics['loss']
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                    self.model.save_pretrained(self.hf_dir+"_best"); self.config.model_config.save_pretrained(self.hf_dir+"_best")
                tqdm.write(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                           f"Test Loss: {test_metrics['loss']:.4f}, "
                           f"Test Acc: {test_metrics['accuracy']:.2f}%")
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'last_model.pth'))
        self.model.save_pretrained(self.hf_dir+"_last"); self.config.model_config.save_pretrained(self.hf_dir+"_last")
        pd.DataFrame(self.results).to_csv(os.path.join(self.save_dir, 'training_results.csv'), index=False)
        if not(self.hf_token is None) : 
            push_model_to_hub(self.repo_name, self.hf_dir+"_best", self.hf_token, 
                              subfolder=self.hf_dir.split(os.path.sep)[-1]+"_best")
            push_model_to_hub(self.repo_name, self.hf_dir+"_last", self.hf_token, 
                              subfolder=self.hf_dir.split(os.path.sep)[-1]+"_last")
        print("Training completed. Results and models saved.")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _evaluate(self, data_loader, max_samples=None):
        self.model.eval()
        total_loss = 0
        all_targets, all_predictions = [], []
        samples_processed = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                samples_processed += targets.size(0)
                if max_samples and samples_processed >= max_samples:
                    break
        avg_loss = total_loss / (samples_processed // data_loader.batch_size)
        accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}