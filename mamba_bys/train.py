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

class Trainer:
    def __init__(self, model, dataset, batch_size, learning_rate, num_epochs, save_dir='models'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.compile(model.to(self.device))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {avg_loss:.4f}")
            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1: # every 5 epoch
                self.evaluate()
                self.save_checkpoint(epoch)

    def evaluate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']