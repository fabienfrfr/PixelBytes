#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien

https://www.perplexity.ai/search/tu-peux-me-dire-combien-obtien-z6lt6R6IRrmwnILyCqwZxw
"""

import torch
import torch.nn.functional as F

class SequenceGenerator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.tokenizer = config.tokenizer
        self.newline_token = self.tokenizer.token_to_id[b'\n']
        self.tab_token = self.tokenizer.token_to_id[b'\t']

    def generate(self, start_sequence, max_length=100, temperature=1.0):
        self.model.eval()
        current_sequence = torch.tensor(start_sequence).unsqueeze(0)
        generated = []
        is_pixel_mode = False

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(current_sequence).squeeze()
                probs = F.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if is_pixel_mode and self.is_text_token(next_token_id):
                    generated.append(self.newline_token)
                    is_pixel_mode = False

                generated.append(next_token_id)

                if next_token_id == self.newline_token:
                    is_pixel_mode = not is_pixel_mode
                elif next_token_id == self.tab_token and is_pixel_mode:
                    is_pixel_mode = True
                elif not self.is_text_token(next_token_id):
                    is_pixel_mode = True

                current_sequence = self.update_sequence(current_sequence, next_token_id)

        return self.linear_to_2d(generated)

    def is_text_token(self, token_id):
        return 32 <= token_id <= 126

    def update_sequence(self, sequence, new_token_id):
        return torch.cat([sequence[:, 1:], torch.tensor([[[[0,0,0], [0,new_token_id,0], [0,0,0]]]]), dim=1)

    def linear_to_2d(self, sequence):
        result = []
        current_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        is_pixel_mode = False
        pixel_pos = 0

        for token_id in sequence:
            if token_id in [self.newline_token, self.tab_token]:
                if current_matrix != [[0, 0, 0], [0, 0, 0], [0, 0, 0]]:
                    result.append(current_matrix)
                result.append([[0, 0, 0], [0, token_id, 0], [0, 0, 0]])
                current_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                if token_id == self.newline_token:
                    is_pixel_mode = not is_pixel_mode
                pixel_pos = 0
            elif not is_pixel_mode or self.is_text_token(token_id):
                current_matrix[1][1] = token_id
                result.append(current_matrix)
                current_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            else:  # Pixel mode
                current_matrix[pixel_pos // 3][pixel_pos % 3] = token_id
                pixel_pos += 1
                if pixel_pos == 9:
                    result.append(current_matrix)
                    current_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    pixel_pos = 0

        if current_matrix != [[0, 0, 0], [0, 0, 0], [0, 0, 0]]:
            result.append(current_matrix)

        return torch.tensor(result)
