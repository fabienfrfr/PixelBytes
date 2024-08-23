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

    def train_and_evaluate(self):
        best_loss = float('inf')
        test_loss, accuracy, f1 = self._evaluate() # random output
        self.results.append({'epoch': 0, 'train_loss': test_loss, 'test_loss': test_loss, 'accuracy': accuracy, 'f1_score': f1})
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            if (epoch + 1) % self.eval_every == 0 or epoch == self.num_epochs - 1:
                test_loss, accuracy, f1 = self._evaluate()
                self.results.append({'epoch': epoch + 1, 'train_loss': train_loss, 'test_loss': test_loss, 'accuracy': accuracy, 'f1_score': f1})
                if test_loss < best_loss:
                    best_loss = test_loss
                    self._save_model('best_model.pth')
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}")
        # Final evaluation
        if self.num_epochs % self.eval_every != 0:  # Évaluer une dernière fois si pas fait
            test_loss, accuracy, f1 = self._evaluate()
            self.results.append({'epoch': self.num_epochs, 'train_loss': train_loss, 'test_loss': test_loss, 'accuracy': accuracy, 'f1_score': f1})
            print(f"Final Evaluation - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.4f}")
        self._save_results()

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, targets in tqdm(self.train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_targets, all_predictions = [], []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.4f}")
        return avg_loss, accuracy, f1

    def _save_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def _save_results(self):
        pd.DataFrame(self.results).to_csv(os.path.join(self.save_dir, 'training_results.csv'), index=False)
        print("Training results saved.")