#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien
In dev.. (process_token not really working)
https://www.perplexity.ai/search/tu-peux-me-dire-combien-obtien-z6lt6R6IRrmwnILyCqwZxw
"""
import torch
import torch.nn.functional as F

class SequenceGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.newline_token = self.tokenizer.vocab[b'\n']
        self.tab_token = self.tokenizer.vocab[b'\t']

    def generate_complete(self, start_sequence, max_length=100, temperature=1.0):
        current_sequence, sequence_clock = self.ids_to_matrices(start_sequence)
        generated_sequence = current_sequence.clone()
        
        for _ in range(max_length):
            next_token_id = self._generate_next_token(current_sequence, temperature)
            next_matrix, sequence_clock = self._process_token(next_token_id, current_sequence, sequence_clock)
            current_sequence = torch.cat([current_sequence[1:], next_matrix.unsqueeze(0)])
            generated_sequence = torch.cat([generated_sequence, next_matrix.unsqueeze(0)])

        return generated_sequence.cpu().numpy()

    def generate_stream(self, start_sequence, max_length=100, temperature=1.0):
        current_sequence, sequence_clock = self.ids_to_matrices(start_sequence)
        for matrix in current_sequence:
            yield matrix.cpu().numpy()

        for _ in range(max_length):
            next_token_id = self._generate_next_token(current_sequence, temperature)
            next_matrix, sequence_clock = self._process_token(next_token_id, current_sequence, sequence_clock)
            current_sequence = torch.cat([current_sequence[1:], next_matrix.unsqueeze(0)])
            yield next_matrix.cpu().numpy()

    def _generate_next_token(self, current_sequence, temperature):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(current_sequence.unsqueeze(0)).squeeze()
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()

    def _process_token(self, token_id, seq, sequence_clock):
        token = self.tokenizer.ids_to_tokens[token_id]
        if len(seq) == 0:
            return torch.tensor([[0,0,0], [0, token_id, 0], [0,0,0]]), [0, 0]
    
        prev_prev_was_pixel = isinstance(self.tokenizer.ids_to_tokens[seq[-2][1, 1].item()], tuple) if len(seq) > 1 else False
        prev_token_id = seq[-1][1, 1].item()
        prev_token = self.tokenizer.ids_to_tokens[prev_token_id]
    
        # Calculer l'index i pour la référence à la séquence précédente
        a, b = sequence_clock[-2:]  # période
        i, gap = a + len(seq) - b - 1, b - a  # -1 pour ajuster l'index (0-based)
    
        def safe_get(seq, index,default=0):
            return seq[index][1,1].item() if 0 <= index < len(seq) else default
    
        if isinstance(token, bytes):
            if token in [b'\n', b'\t']:
                if isinstance(prev_token, tuple):  # Si le précédent était un pixel
                    if gap == 1: matrix = torch.tensor([[0, 0, 0], [prev_token_id, token_id, 0], [0,0,0]])
                    else : matrix = torch.tensor([[safe_get(seq, i), safe_get(seq, i+1), 0],
                                           [prev_token_id, token_id, 0], [0,0,0]])
                else:
                    matrix = torch.tensor([[0,0,0], [prev_token_id, token_id, 0], [0,0,0]])
            else:
                matrix = torch.tensor([[0,0,0], [0 if prev_prev_was_pixel else prev_token_id, token_id, 0], [0,0,0]])
            sequence_clock.append(len(seq))  # Mise à jour du clock pour tous les bytes
        elif isinstance(token, tuple):  # Pixel token
            if prev_token == b'\t' and gap > 1:
                matrix = torch.tensor([[0, safe_get(seq, i+1), safe_get(seq, i+2)],
                                       [0, token_id, 0], [0,0,0]])
            elif isinstance(prev_token, bytes):
                matrix = torch.tensor([[0, 0, 0], [0, token_id, 0], [0,0,0]]) # warning here : change following bytes token, not \n, can cause problem during next generation
                if prev_token != b'\n' : sequence_clock.append(len(seq))
            elif gap <= 3:
                matrix = torch.tensor([[0, 0, 0], [prev_token_id, token_id, 0], [0,0,0]])
            else:
                matrix = torch.tensor([[safe_get(seq, i), safe_get(seq, i+1), safe_get(seq, i+2)],
                                       [prev_token_id, token_id, 0],
                                       [0,0,0]])
        return matrix, sequence_clock

    def ids_to_matrices(self, sequence):
        matrices = []
        sequence_clock = [0, 0]
        for token_id in sequence:
            matrix, sequence_clock = self._process_token(token_id, matrices, sequence_clock)
            matrices.append(matrix)
        return torch.stack(matrices), sequence_clock