#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

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
    def __init__(self, config: EvaluateMetricConfig, data: np.ndarray, tokenizer):
        self.config = config
        self.data = data
        self.model = None
        self.device = torch.device(config.device)
        self.sequence_generator = SequenceGenerator(tokenizer)
        self.df = None
        os.makedirs(self.config.output_dir, exist_ok=True)

    def reset(self, model: torch.nn.Module):
        self.model = model.to(self.device).eval()

    def _generate_sequence(self, input_seq: np.ndarray) -> np.ndarray:
        self.sequence_generator.reset(input_seq)
        input_tensor = torch.from_numpy(input_seq).float().to(self.device)
        
        with torch.no_grad():
            for _ in range(self.config.generation_length):
                output = self.model(input_tensor.unsqueeze(0))
                next_token = output.squeeze().cpu().numpy()
                self.sequence_generator.update_sequence(next_token)
                input_tensor = torch.from_numpy(self.sequence_generator.sequence[-self.config.input_seq_length:]).float().to(self.device)

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
                results.append({
                    "row": row_idx,
                    "n_seq": seq_idx,
                    "model": model_name,
                    "hamming": hamming(generated_seq.flatten(), target_seq.flatten()),
                    "cosine": cosine(generated_seq.flatten(), target_seq.flatten()),
                    "bleu": sentence_bleu([list(map(str, target_seq.flatten()))], list(map(str, generated_seq.flatten())))
                })

        self.df = pd.DataFrame(results)
        self.df.to_csv(os.path.join(self.config.output_dir, f"{model_name}.csv"), index=False)
        return self.df
"""
# Usage
config = EvaluateMetricConfig()
data = np.random.rand(10, 100, 3, 3)  # Example data
tokenizer = YourTokenizer()  # Provide your tokenizer here
evaluator = EvaluateMetric(config, data, tokenizer)

model = YourModel()  # Your actual model here
evaluator.reset(model)
results = evaluator.evaluate("model_name")
"""