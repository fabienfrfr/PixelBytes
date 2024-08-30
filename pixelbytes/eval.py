#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

from .generator import SequenceGenerator

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial.distance import hamming, cosine
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import os

@dataclass
class EvaluateMetricConfig:
    input_seq_length: int = 64
    generation_length: int = 16
    output_dir: str = "eval_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class EvaluateMetric:
    def __init__(self, config: EvaluateMetricConfig, data_pxby, tokenizer):
        self.config = config
        self.data = [np.array(d) for d in data_pxby] 
        self.model = None
        self.device = torch.device(config.device)
        self.sequence_generator = SequenceGenerator(tokenizer)
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.df = None

    def reset(self, model: torch.nn.Module):
        self.model = model.to(self.device).eval()

    def _generate_sequence(self, input_seq: np.ndarray) -> np.ndarray:
        self.sequence_generator.reset(input_seq)
        input_tensor = torch.from_numpy(input_seq).long().to(self.device)

        with torch.no_grad():
            for _ in range(self.config.generation_length):
                output = self.model(input_tensor.unsqueeze(0))
                _, predicted = output.max(1)
                next_token_id = predicted.squeeze().cpu().numpy().item() # ensure is not numpy object
                self.sequence_generator.update_sequence(next_token_id)
                input_tensor = torch.from_numpy(self.sequence_generator.sequence).long().to(self.device)
        return self.sequence_generator.sequence[-self.config.generation_length:]

    def evaluate(self, model_name: str) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not set. Call reset() with a model before evaluating.")

        results = []
        stride = self.config.input_seq_length + self.config.generation_length

        for row_idx, sequence in enumerate(tqdm(self.data, desc=f"Evaluating {model_name}")):
            L = len(sequence)
            starts = np.arange(0, L, stride)
            indices = (starts[:, None] + np.arange(stride)) % L
            sub_seqs = sequence[indices]
            inputs = sub_seqs[:, :self.config.input_seq_length]
            targets = sub_seqs[:, self.config.input_seq_length:]

            for seq_idx, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
                generated_seq = self._generate_sequence(input_seq)
                
                target_flat = target_seq.flatten()
                generated_flat = generated_seq.flatten()
                
                results.append({
                    "row": row_idx,
                    "n_seq": seq_idx,
                    "hamming": hamming(generated_flat, target_flat),
                    "cosine": 1 - cosine(generated_flat, target_flat),
                    "bleu": sentence_bleu([list(map(str, target_flat))], list(map(str, generated_flat))),
                })

        self.df = pd.DataFrame(results)
        self.df.to_csv(os.path.join(self.config.output_dir, f"{model_name}.csv"), index=False)
        return self.df
