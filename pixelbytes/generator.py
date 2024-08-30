#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien
"""
import numpy as np

class SequenceGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.newline_token = self.tokenizer.vocab[b'\n']
        self.tab_token = self.tokenizer.vocab[b'\t']
        self.sequence = None
        self.clock = None

    def reset(self, start_sequence, line_period=64):
        # shape : start_sequence : (L>50, M=3, N=3) : long --> input of model
        L, M, N = start_sequence.shape
        if L < line_period : print(f"Input sequence dimensions should be > 2*25 for image minimum, but got {L}, choose 64 (images line)")
        # clone sequence and find clock
        self.sequence = start_sequence.copy()
        self.clock = self._find_clock(self.sequence)

    def update_sequence(self, tokens_ids):
        # shape : (1) --> output of model
        if not(hasattr(tokens_ids, '__iter__')):
            tokens_ids = [tokens_ids]
        for token_id in tokens_ids :
            next_matrix, self.clock = self._process_token(token_id, self.sequence, self.clock)
            self.sequence = np.concatenate([self.sequence, next_matrix], axis=0)
        return self.sequence #not optimal for RNN or SSM for stepping ?
    
    # intern process
    def _find_clock(self, sequence):
        sequence_clock = [0, 0]
        clock = 0
        for ids in range(len(sequence)):
            token = self.tokenizer.ids_to_tokens[sequence[ids,1,1]]
            if isinstance(token, bytes) : 
                sequence_clock = sequence_clock + [ids] if clock > 0 else sequence_clock + 2*[ids]
                clock += 1
        return sequence_clock
    
    def _process_token(self, token_id, seq, sequence_clock):
        token = self.tokenizer.ids_to_tokens[token_id]
        if len(seq) == 0: # warning here (if you start with a tuple, you can have issues, but with text its ok !)
            return np.array([[0,0,0], [0, token_id, 0], [0,0,0]])[None], [0, 0]
        
        prev_prev_was_pixel = isinstance(self.tokenizer.ids_to_tokens[seq[-2][1, 1].item()], tuple) if len(seq) > 1 else False
        prev_token_id = seq[-1][1, 1].item()
        prev_token = self.tokenizer.ids_to_tokens[prev_token_id]
    
        # index calculation for sequence ref
        a, b = sequence_clock[-2:]  # step
        i, gap = a + len(seq) - b - 1, b - a  # (0-based)
        def safe_get(seq, index,default=0):
            return seq[index][1,1].item() if 0 <= index < len(seq) else default
    
        if isinstance(token, bytes):
            if token in [b'\n', b'\t']:
                if isinstance(prev_token, tuple):  # if pixel
                    if gap == 1: matrix = np.array([[0, 0, 0], [prev_token_id, token_id, 0], [0,0,0]])
                    else : matrix = np.array([[safe_get(seq, i), safe_get(seq, i+1), 0],
                                           [prev_token_id, token_id, 0], [0,0,0]])
                else:
                    matrix = np.array([[0,0,0], [prev_token_id, token_id, 0], [0,0,0]])
            else:
                matrix = np.array([[0,0,0], [0 if prev_prev_was_pixel else prev_token_id, token_id, 0], [0,0,0]])
            sequence_clock.append(len(seq))  # update clock whatever bytes
        elif isinstance(token, tuple):  # Pixel token
            if prev_token == b'\t' and gap > 1:
                matrix = np.array([[0, safe_get(seq, i+1), safe_get(seq, i+2)],
                                       [0, token_id, 0], [0,0,0]])
            elif isinstance(prev_token, bytes):
                matrix = np.array([[0, 0, 0], [0, token_id, 0], [0,0,0]]) # warning here : change following bytes token, not \n, can cause problem during next generation
                if prev_token != b'\n' : sequence_clock.append(len(seq))
            elif len(seq) - b > gap : # line2 > line 1
                matrix = np.array([[0, 0, 0], [prev_token_id, token_id, 0], [0,0,0]])
            else:
                matrix = np.array([[safe_get(seq, i), safe_get(seq, i+1), safe_get(seq, i+2)],
                                       [prev_token_id, token_id, 0],
                                       [0,0,0]])
        return matrix[None], sequence_clock


