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
import os, json, csv

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, batch_size, learning_rate, num_epochs, 
                 save_dir='models', compile_model=True, model_name="unnamed", dataset_name="unnamed", eval_every=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.eval_every = eval_every
        
        if compile_model and torch.__version__ >= "2.0.0":
            print("Compiling model...")
            self.model = torch.compile(self.model)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.save_dir = os.path.join(save_dir, f"{self.model_name}_{self.dataset_name}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []

        self.save_metadata()

    def save_metadata(self):
        metadata = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "num_epochs": self.num_epochs,
            "batch_size": self.train_loader.batch_size,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "device": str(self.device),
            "eval_every": self.eval_every
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    def train(self):
        best_accuracy = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
            
            if (epoch + 1) % self.eval_every == 0 or epoch == self.num_epochs - 1:
                accuracy, avg_test_loss = self.evaluate()
                self.test_losses.append(avg_test_loss)
                self.test_accuracies.append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.save_checkpoint(epoch)

                self.save_training_progress()

        print(f"Best Test Accuracy: {best_accuracy:.2f}%")
        self.save_training_progress(final=True)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.test_loader)
        print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}')
        return accuracy, avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if is_best:
            path = os.path.join(self.save_dir, f'best_model_{self.model_name}_{self.dataset_name}.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}_{self.model_name}_{self.dataset_name}.pth')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved for epoch {epoch+1}" + (" (Best Model)" if is_best else ""))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def save_training_progress(self, final=False):
        filename = 'training_progress.csv' if not final else 'final_training_progress.csv'
        with open(os.path.join(self.save_dir, filename), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
            for epoch, train_loss in enumerate(self.train_losses, 1):
                test_loss = self.test_losses[epoch // self.eval_every - 1] if epoch % self.eval_every == 0 else ""
                test_acc = self.test_accuracies[epoch // self.eval_every - 1] if epoch % self.eval_every == 0 else ""
                writer.writerow([epoch, train_loss, test_loss, test_acc])


        print(f"Best Test Accuracy: {best_accuracy:.2f}%")
        self.save_training_progress(final=True)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.test_loader)
        print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}')
        return accuracy, avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if is_best:
            path = os.path.join(self.save_dir, f'best_model_{self.model_name}_{self.dataset_name}.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}_{self.model_name}_{self.dataset_name}.pth')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved for epoch {epoch+1}" + (" (Best Model)" if is_best else ""))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def save_training_progress(self, final=False):
        filename = 'training_progress.csv' if not final else 'final_training_progress.csv'
        with open(os.path.join(self.save_dir, filename), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
            for epoch, (train_loss, test_loss, test_acc) in enumerate(zip(self.train_losses, self.test_losses, self.test_accuracies), 1):
                writer.writerow([epoch, train_loss, test_loss, test_acc])


