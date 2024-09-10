#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
from .generator import SequenceGenerator
import torch

class Displays:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sequence_generator = SequenceGenerator(self.tokenizer)
        self._images = None #previous img
        self._texts = None #previous text
        self.fig = None
    
    def reset(self, model: torch.nn.Module):
        self.model = model.eval()
    
    # specific process
    def generate_sequence(self, complete_sequence, start_length=64, window=None, gen_window=None):
        L, _, _ = complete_sequence.shape
        input_seq = complete_sequence[:start_length]
        self.sequence_generator.reset(input_seq)
        is_progressive = window is not None and gen_window is not None
        while start_length < L :
            input_tensor = torch.from_numpy(self.sequence_generator.sequence).long()
            if is_progressive:
                update_length = min(window, L - len(self.sequence_generator.sequence) - gen_window)
                update_token_id = complete_sequence[start_length:start_length+update_length][:,1,1]
                new_sequence = self.sequence_generator.update_sequence(update_token_id)
                gen_length = min(gen_window, L - len(self.sequence_generator.sequence))
            else: gen_length = L - len(self.sequence_generator.sequence)
            with torch.no_grad():
                for _ in range(gen_length):
                    output = self.model(input_tensor.unsqueeze(0))
                    _, predicted = output.max(1)
                    next_token_id = predicted.squeeze().cpu().numpy().item()
                    new_sequence = self.sequence_generator.update_sequence(next_token_id)
                    input_tensor = torch.from_numpy(new_sequence).long()
            start_length = len(self.sequence_generator.sequence)
        return self.sequence_generator.sequence

    def process_sequence(self, complete_sequence, start_length=64, window=None, gen_window=None):
        final_sequence = self.generate_sequence(complete_sequence, start_length, window, gen_window)
        tokens_arr_obj = self.tokenizer.convert_ids_to_tokens(final_sequence[:,1,1])
        images = self.reconstruct_imgs(tokens_arr_obj)
        text_blocks = self.reconstruct_text(tokens_arr_obj)
        return images, text_blocks

    def reconstruct_imgs(self, tokens_arr, min_row_length=3, max_gap=50):
        img_idx = [i for i, x in enumerate(tokens_arr) if isinstance(x, tuple) and len(x) == 3]
        if len(img_idx) < min_row_length: return []
        
        img_idx = np.array(img_idx)
        big_gaps = np.where(np.diff(img_idx) > max_gap)[0] + 1
        image_splits = np.split(img_idx, big_gaps)
        images = []
        for split in image_splits:
            if len(split) < min_row_length:
                continue
            row_breaks = np.concatenate(([0], np.where(np.diff(split) > 1)[0] + 1, [len(split)]))
            row_lengths = np.diff(row_breaks)
            valid_rows = row_lengths >= min_row_length
            if not np.any(valid_rows):
                continue
            most_common_length = np.bincount(row_lengths[valid_rows]).argmax()
            num_rows = np.sum(valid_rows)
            img = np.full((num_rows, most_common_length, 3), 255, dtype=np.uint8)
            valid_indices = split[np.concatenate([np.arange(start, end) for start, end, valid 
                                                  in zip(row_breaks[:-1], row_breaks[1:], valid_rows) 
                                                  if valid])]
            img_flat = img.reshape(-1, 3)
            for i, idx in enumerate(valid_indices):
                if i < len(img_flat):
                    img_flat[i] = tokens_arr[idx]
            images.append({'image': img, 'start_index': split[0], 'end_index': split[-1]})
        self._images = images
        return images

    def reconstruct_text(self, tokens_arr, min_length=10, max_gap=10):
        text_idx = [i for i, x in enumerate(tokens_arr) if isinstance(x, bytes)]
        if len(text_idx) < min_length: return []
        text_idx = np.array(text_idx)
        big_gaps = np.where(np.diff(text_idx) > max_gap)[0] + 1
        text_splits = np.split(text_idx, big_gaps)
        text_blocks = [{ 'text': b"".join(tokens_arr[i] for i in split),
                        'start_index': split[0],'end_index': split[-1]}
                       for split in text_splits if len(split) >= min_length]
        self._texts = text_blocks
        return text_blocks

    def show(self, images, text_blocks, max_text_length=300):
        # Create figure with fixed size
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        # Image subplot
        ax_images = plt.subplot(gs[0])
        ax_images.set_title("Generated Images")
        ax_images.axis('off')
        
        # Calculate grid size for images
        n_images = len(images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        for i, img_data in enumerate(images):
            ax = fig.add_subplot(grid_size, grid_size, i+1)
            ax.imshow(img_data['image'])
            ax.axis('off')
        
        # Text subplot
        ax_text = plt.subplot(gs[1])
        ax_text.set_title("Generated Text")
        ax_text.axis('off')
        
        # Combine all text blocks
        combined_text = ""
        for text_data in text_blocks:
            text = text_data['text'].decode('utf-8', errors='replace')[:max_text_length]
            if len(text_data['text']) > max_text_length:
                text += "..."
            combined_text += text + "\n\n"  # Add two newlines between text blocks
        
        # Display combined text
        ax_text.text(0.05, 0.95, combined_text, va='top', ha='left', wrap=True, 
                     fontsize=10)
        
        plt.tight_layout()
        plt.savefig('image.svg'); plt.savefig('image.png', dpi=300)
        plt.show()
        plt.close()

