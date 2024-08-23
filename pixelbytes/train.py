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

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, batch_size, learning_rate, num_epochs, 
                 save_dir='models', model_name="model", dataset_name="dataset", eval_every=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.save_dir = os.path.join(save_dir, f"{model_name}_{dataset_name}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.eval_every = eval_every
        self.results = []
        self.test_size = len(test_dataset)

    def train_and_evaluate(self):
        best_test_loss = float('inf')
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            train_loss = self._train_epoch()
            
            if (epoch + 1) % self.eval_every == 0 or epoch == self.num_epochs - 1:
                train_metrics = self._evaluate(self.train_loader, max_samples=self.test_size)
                test_metrics = self._evaluate(self.test_loader)
                self.results.append({
                    'epoch': epoch + 1, 
                    'train_loss': train_loss,
                    'train_eval_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'test_loss': test_metrics['loss'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1']})
                
                if test_metrics['loss'] < best_test_loss:
                    best_test_loss = test_metrics['loss']
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                tqdm.write(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                           f"Test Loss: {test_metrics['loss']:.4f}, "
                           f"Test Acc: {test_metrics['accuracy']:.2f}%")

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'last_model.pth'))
        pd.DataFrame(self.results).to_csv(os.path.join(self.save_dir, 'training_results.csv'), index=False)
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