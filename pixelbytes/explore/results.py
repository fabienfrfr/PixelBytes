#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfr
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_csv_files(directory, separators=['rnn', 'ssm', 'attention']):
    data_dict = {sep: [] for sep in separators}
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            model_name = os.path.splitext(file)[0]
            file_path = os.path.join(directory, file)
            # Determine the model group
            group = next((sep for sep in separators if sep in model_name.lower()), 'other')
            # Load the data
            df = pd.read_csv(file_path)
            df['model'] = model_name  # Add model name to the dataframe
            # Add the dataframe to the appropriate list
            data_dict[group].append(df)
    # Concatenate dataframes for each group
    for group in data_dict:
        if data_dict[group]: data_dict[group] = pd.concat(data_dict[group], ignore_index=True)
        else: data_dict[group] = pd.DataFrame()  # Empty dataframe if no data for this group
    return data_dict

def plot_training_results(data, save_folder=""):
    plt.rcParams.update({'font.size': 16}) 
    metrics = [('loss', 'train_eval_loss', 'test_loss', 'Loss'),
               ('accuracy', 'train_accuracy', 'test_accuracy', 'Accuracy (%)'),]
               #('f1', 'train_f1', 'test_f1', 'F1 Score')]
    #fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='white')
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 6), facecolor='white')
    
    colors = {'rnn': 'blue', 'ssm': 'green', 'attention': 'red'}
    
    for idx, (metric, train_col, test_col, ylabel) in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor('white')  # Set the background of each subplot to white
        
        max_epoch = 0
        for group, df in data.items():
            if not df.empty:
                sns.lineplot(data=df, x='epoch', y=train_col, ax=ax, label=f'{group.upper()} Train', 
                             color=colors[group], errorbar='sd', linewidth=2)
                sns.lineplot(data=df, x='epoch', y=test_col, ax=ax, label=f'{group.upper()} Test', 
                             color=colors[group], linestyle='--', errorbar='sd', linewidth=2)
                max_epoch = max(max_epoch, df['epoch'].max())
        
        ax.set_xlim(0, max_epoch)  # Set x-axis limits based on data
        ax.set_ylim(0)  # Set y-axis to start from 0
        ax.set_xlabel('Epoch', fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Add a more visible grid
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        
        # Add black border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        
        # Remove legend for the first two subplots
        if idx < 2:
            ax.get_legend().remove()
    
    # Add a single legend to the last subplot
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    
    plt.tight_layout()
    
    # Create save folder if it doesn't exist
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        
        # Save figures
        save_path = os.path.join(save_folder, 'training_results')
        plt.savefig(save_path + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    
    plt.show()
    plt.close()
    