#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class SequenceReconstructor:
    def __init__(self, tokenizer, min_row_length=3):
        self.tokenizer = tokenizer
        self.min_row_length = min_row_length
        self.elements = []
        self.current = {'type': None, 'data': []}

    def process(self, token_id):
        token = self.tokenizer.ids_to_tokens[token_id]
        
        if isinstance(token, tuple) and len(token) == 3:
            self._process_element('image', token)
        elif isinstance(token, bytes):
            self._process_element('text', token)

    def _process_element(self, type, data):
        if self.current['type'] != type:
            self._flush_current()
            self.current['type'] = type
        self.current['data'].append(data)
        
        if type == 'image' and len(self.current['data']) >= self.min_row_length ** 2:
            self._flush_current()

    def _flush_current(self):
        if self.current['type'] == 'image':
            rows = [self.current['data'][i:i+self.min_row_length] 
                    for i in range(0, len(self.current['data']), self.min_row_length)]
            if len(rows) >= self.min_row_length:
                self.elements.append(('image', np.array(rows[:self.min_row_length], dtype=np.uint8)))
                self.current['data'] = self.current['data'][self.min_row_length ** 2:]
            else:
                self.elements.append(('image', np.array(rows, dtype=np.uint8)))
                self.current['data'] = []
        elif self.current['type'] == 'text':
            text = b''.join(self.current['data']).decode('utf-8', errors='ignore')
            if text.strip():
                self.elements.append(('text', text))
            self.current['data'] = []
        self.current['type'] = None

    def get_result(self):
        self._flush_current()
        return self.elements

def display_result(result, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    y_pos = 1
    for type, data in result:
        if type == 'image':
            ax.imshow(data, extent=[0.1, 0.3, y_pos - 0.2, y_pos], aspect='auto')
            y_pos -= 0.25
        elif type == 'text':
            ax.text(0.1, y_pos, data, va='top', fontsize=8, wrap=True)
            y_pos -= 0.07
    plt.tight_layout()
    plt.show()