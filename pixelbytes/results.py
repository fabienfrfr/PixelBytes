#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Configuration du style (inchangée)
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_csv_files(directory_or_file):
    data_dict = {}
    if os.path.isdir(directory_or_file):
        for file in os.listdir(directory_or_file):
            if file.endswith('.csv'):
                model_name = os.path.splitext(file)[0]
                file_path = os.path.join(directory_or_file, file)
                data_dict[model_name] = pd.read_csv(file_path)
    elif os.path.isfile(directory_or_file) and directory_or_file.endswith('.csv'):
        df = pd.read_csv(directory_or_file)
        if 'model' in df.columns:
            for model in df['model'].unique():
                data_dict[model] = df[df['model'] == model].reset_index(drop=True)
        else:
            data_dict['unnamed_model'] = df
    return data_dict

def save_plot(fig, base_name):
    for ext in ['svg', 'png']:
        fig.savefig(f"{base_name}.{ext}", format=ext, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_training_results(train_dfs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    metrics = {
        'loss': ['train_eval_loss', 'test_loss'],
        'accuracy': ['train_accuracy', 'test_accuracy'],
        'f1': ['train_f1', 'test_f1']
    }
    
    for metric_name, columns in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, df in train_dfs.items():
            for col in columns:
                if col in df.columns:
                    sns.lineplot(data=df, x='epoch', y=col, ax=ax, label=f'{model_name}_{col}', linewidth=2, marker='o', markersize=4)
        ax.set_title(f'{metric_name.capitalize()} over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_plot(fig, os.path.join(save_dir, f'{metric_name}_comparison'))

def plot_evaluation_metrics(eval_dfs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    all_metrics = set()
    for df in eval_dfs.values():
        all_metrics.update([col for col in df.columns if col not in ['model', 'generated']])
    
    for metric in all_metrics:
        if metric.endswith(('_min', '_avg', '_max')):
            base_metric = metric.rsplit('_', 1)[0]
            fig, ax = plt.subplots(figsize=(10, 6))
            data = []
            for model_name, df in eval_dfs.items():
                if f"{base_metric}_min" in df.columns and f"{base_metric}_max" in df.columns:
                    data.append({
                        'model': model_name,
                        'min': df[f"{base_metric}_min"].iloc[0],
                        'avg': df[f"{base_metric}_avg"].iloc[0],
                        'max': df[f"{base_metric}_max"].iloc[0]
                    })
            if data:
                plot_df = pd.DataFrame(data)
                sns.barplot(x='model', y='avg', data=plot_df, ax=ax)
                ax.errorbar(x=range(len(plot_df)), y=plot_df['avg'], 
                            yerr=[plot_df['avg'] - plot_df['min'], plot_df['max'] - plot_df['avg']],
                            fmt='none', c='black', capsize=5)
                ax.set_title(f'{base_metric.capitalize()} Comparison')
                ax.set_xlabel('Model')
                ax.set_ylabel('Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                save_plot(fig, os.path.join(save_dir, f'{base_metric}_comparison'))
        elif not any(metric.endswith(suffix) for suffix in ('_min', '_avg', '_max')):
            fig, ax = plt.subplots(figsize=(10, 6))
            for model_name, df in eval_dfs.items():
                if metric in df.columns:
                    sns.barplot(x=[model_name], y=[df[metric].iloc[0]], ax=ax)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_plot(fig, os.path.join(save_dir, f'{metric}_comparison'))

def main(train_results_path, eval_metrics_path):
    train_dfs = load_csv_files(train_results_path)
    eval_dfs = load_csv_files(eval_metrics_path)
    
    plot_training_results(train_dfs, 'training_plots')
    plot_evaluation_metrics(eval_dfs, 'nlp_metric_plots')

if __name__ == '__main__':
    train_results_path = 'path/to/your/train_results_directory_or_file'
    eval_metrics_path = 'path/to/your/eval_metrics_directory_or_file'
    main(train_results_path, eval_metrics_path)