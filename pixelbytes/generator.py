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

    def generate(self, start_sequence, max_length=100, temperature=1.0):
        self.model.eval()
        current_sequence = torch.tensor(start_sequence).unsqueeze(0)
        is_text_mode = True

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(current_sequence).squeeze()
                probs = F.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                yield next_token_id

                if next_token_id == self.config.eos_token_id:
                    break

                if next_token_id == self.config.mode_switch_token_id:
                    is_text_mode = not is_text_mode

                current_sequence = self._update_sequence(current_sequence, next_token_id, is_text_mode)

    def _update_sequence(self, sequence, new_token_id, is_text_mode):
        if is_text_mode:
            sequence[0, -1, 1, 1:] = torch.tensor([sequence[0, -1, 1, 0], new_token_id])
        else:
            new_matrix = torch.zeros(1, 1, 3, 3, dtype=torch.long)
            new_matrix[0, 0, 1, 1] = new_token_id
            sequence = torch.cat([sequence[:, 1:], new_matrix], dim=1)
        return sequence
