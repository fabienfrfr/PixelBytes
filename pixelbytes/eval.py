#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
from scipy.spatial.distance import hamming, cosine
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
from tqdm import tqdm

@dataclass
class EvaluateMetricConfig:
    metrics: List[str] = ("edit", "hamming", "cosine", "bleu", "diversity", "perplexity")
    batch_size: int = 32
    num_input_sequences: int = 100  # Nombre de séquences d'entrée à utiliser
    sequence_length: int = 50
    vocab: str = None  # Sera déterminé automatiquement
    output_file: str = "evaluation_results.csv"

class EvaluateMetric:
    def __init__(self, config: EvaluateMetricConfig, models: Dict[str, torch.nn.Module], dataloader: torch.utils.data.DataLoader):
        self.config = config
        self.models = models
        self.dataloader = dataloader
        self.vocab = self._determine_vocab()
        self.metric_funcs = {
            "edit": self._edit_distance,
            "hamming": self._hamming_distance,
            "cosine": self._cosine_distance,
            "bleu": self._bleu_score,
            "diversity": self._diversity_score,
            "perplexity": self._perplexity_score
        }

    def _determine_vocab(self):
        # Déterminer le vocabulaire à partir d'un échantillon de données
        sample = next(iter(self.dataloader))
        if isinstance(sample, torch.Tensor):
            sample = self._tensor_to_strings(sample)
        return ''.join(sorted(set(''.join(sample))))

    def evaluate(self) -> pd.DataFrame:
        input_sequences = self._get_input_sequences()
        results = {}

        for model_name, model in self.models.items():
            print(f"Evaluating model: {model_name}")
            generated_sequences = self._generate_sequences(model, input_sequences)
            model_results = self._evaluate_sequences(generated_sequences, input_sequences)
            results[model_name] = model_results

        df = pd.DataFrame(results).T
        df.to_csv(self.config.output_file)
        print(f"Results saved to {self.config.output_file}")
        return df

    def _get_input_sequences(self) -> List[str]:
        # Obtenir un ensemble de séquences d'entrée du dataloader
        sequences = []
        for batch in self.dataloader:
            if isinstance(batch, torch.Tensor):
                sequences.extend(self._tensor_to_strings(batch))
            else:
                sequences.extend(batch)
            if len(sequences) >= self.config.num_input_sequences:
                break
        return sequences[:self.config.num_input_sequences]

    def _generate_sequences(self, model: torch.nn.Module, input_sequences: List[str]) -> List[str]:
        model.eval()
        generated_sequences = []
        with torch.no_grad():
            for input_seq in tqdm(input_sequences, desc="Generating sequences"):
                hidden = model.init_hidden(1)
                input_tensor = self._string_to_tensor(input_seq)
                generated = []
                for _ in range(self.config.sequence_length):
                    output, hidden = model(input_tensor, hidden)
                    probs = torch.softmax(output.squeeze(), dim=-1)
                    next_item = torch.multinomial(probs, 1)
                    generated.append(next_item)
                    input_tensor = torch.zeros(1, 1, len(self.vocab))
                    input_tensor.scatter_(2, next_item.unsqueeze(1), 1)
                generated_sequences.append(self._tensor_to_string(torch.cat(generated, dim=1)))
        return generated_sequences

    def _evaluate_sequences(self, generated_sequences: List[str], input_sequences: List[str]) -> Dict[str, float]:
        results = {}
        for metric in self.config.metrics:
            if metric == "diversity":
                results[f"{metric}"] = self.metric_funcs[metric](generated_sequences)
            elif metric == "perplexity":
                perplexities = [self.metric_funcs[metric](seq) for seq in generated_sequences]
                results[f"{metric}_avg"] = np.mean(perplexities)
                results[f"{metric}_min"] = np.min(perplexities)
                results[f"{metric}_max"] = np.max(perplexities)
            else:
                scores = [
                    [self.metric_funcs[metric](input_seq, gen_seq) for input_seq in input_sequences]
                    for gen_seq in generated_sequences
                ]
                results[f"{metric}_min"] = np.min(scores)
                results[f"{metric}_avg"] = np.mean(scores)
                results[f"{metric}_max"] = np.max(scores)
        return results

    def _string_to_tensor(self, string: str) -> torch.Tensor:
        return torch.tensor([[self.vocab.index(c) for c in string]])

    def _tensor_to_string(self, tensor: torch.Tensor) -> str:
        return ''.join(self.vocab[i] for i in tensor.squeeze().tolist())

    def _tensor_to_strings(self, tensor: torch.Tensor) -> List[str]:
        return [''.join(self.vocab[i] for i in seq) for seq in tensor.tolist()]

    # Méthodes de calcul des métriques
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
            input_seq = self._string_to_tensor(sequence)
            output, _ = self.model(input_seq)
            loss = torch.nn.functional.cross_entropy(output.view(-1, len(self.vocab)), input_seq.view(-1))
            return torch.exp(loss).item()

# Exemple d'utilisation
if __name__ == "__main__":
    config = EvaluateMetricConfig()
    
    # Définissez vos modèles et votre dataloader ici
    models = {
        "model1": YourModel1(),
        "model2": YourModel2(),
        # Ajoutez autant de modèles que nécessaire
    }
    dataloader = YourDataLoader()
    
    evaluator = EvaluateMetric(config, models, dataloader)
    results_df = evaluator.evaluate()
    print(results_df)