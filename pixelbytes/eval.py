#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from scipy.spatial.distance import hamming, cosine
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
from tqdm import tqdm

@dataclass
class EvaluateMetricConfig:
    metrics: List[str] = ("edit", "hamming", "cosine", "bleu", "diversity", "perplexity")
    batch_size: int = 32
    num_generated_sequences: int = 100
    sequence_length: int = 50
    vocab: str = "ATGC"  # Pour les séquences ADN

class EvaluateMetric:
    def __init__(self, config: EvaluateMetricConfig, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.metric_funcs = {
            "edit": self._edit_distance,
            "hamming": self._hamming_distance,
            "cosine": self._cosine_distance,
            "bleu": self._bleu_score,
            "diversity": self._diversity_score,
            "perplexity": self._perplexity_score
        }

    def evaluate(self) -> pd.DataFrame:
        input_sequences = self._tensor_to_strings(next(iter(self.dataloader)))
        generated_sequences = self._tensor_to_strings(self.generate_sequences())
        
        results = []
        for gen_seq in tqdm(generated_sequences, desc="Evaluating sequences"):
            row = {"generated": gen_seq}
            for metric in self.config.metrics:
                if metric == "diversity":
                    row[f"{metric}"] = self.metric_funcs[metric](generated_sequences)
                elif metric == "perplexity":
                    row[f"{metric}"] = self.metric_funcs[metric](gen_seq)
                else:
                    scores = [self.metric_funcs[metric](input_seq, gen_seq) for input_seq in input_sequences]
                    row[f"{metric}_min"] = np.min(scores)
                    row[f"{metric}_avg"] = np.mean(scores)
            results.append(row)
        return pd.DataFrame(results)

    def generate_sequences(self) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.init_hidden(self.config.num_generated_sequences)
            input_seq = torch.zeros(self.config.num_generated_sequences, 1, len(self.config.vocab))
            generated = []
            for _ in range(self.config.sequence_length):
                output, hidden = self.model(input_seq, hidden)
                probs = torch.softmax(output.squeeze(), dim=-1)
                next_item = torch.multinomial(probs, 1)
                generated.append(next_item)
                input_seq = torch.zeros(self.config.num_generated_sequences, 1, len(self.config.vocab))
                input_seq.scatter_(2, next_item.unsqueeze(1), 1)
        return torch.cat(generated, dim=1)

    def _tensor_to_strings(self, tensor: torch.Tensor) -> List[str]:
        return [''.join(self.config.vocab[i] for i in seq) for seq in tensor.tolist()]

    def _edit_distance(self, seq1: str, seq2: str) -> float:
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / max(len(seq1), len(seq2))

    def _hamming_distance(self, seq1: str, seq2: str) -> float:
        return hamming(list(seq1), list(seq2))

    def _cosine_distance(self, seq1: str, seq2: str) -> float:
        vec1, vec2 = Counter(seq1), Counter(seq2)
        keys = set(vec1.keys()) | set(vec2.keys())
        vec1 = [vec1.get(k, 0) for k in keys]
        vec2 = [vec2.get(k, 0) for k in keys]
        return cosine(vec1, vec2)

    def _bleu_score(self, seq1: str, seq2: str) -> float:
        return sentence_bleu([list(seq1)], list(seq2))

    def _diversity_score(self, sequences: List[str]) -> float:
        unique_sequences = set(sequences)
        return len(unique_sequences) / len(sequences)

    def _perplexity_score(self, sequence: str) -> float:
        self.model.eval()
        with torch.no_grad():
            input_seq = torch.tensor([[self.config.vocab.index(c) for c in sequence]])
            output, _ = self.model(input_seq)
            loss = torch.nn.functional.cross_entropy(output.view(-1, len(self.config.vocab)), input_seq.view(-1))
            return torch.exp(loss).item()

# Exemple d'utilisation
if __name__ == "__main__":
    config = EvaluateMetricConfig()
    
    # Vous devrez définir votre modèle et votre dataloader ici
    model = YourModel()
    dataloader = YourDataLoader()
    
    evaluator = EvaluateMetric(config, model, dataloader)
    results = evaluator.evaluate()
    print(results)