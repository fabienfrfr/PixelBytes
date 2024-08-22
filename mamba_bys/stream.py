#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien

En cours : https://www.perplexity.ai/search/j-ai-un-code-def-reconstruct-i-6a88DXPFSDmyX.D6Elpb0g
"""

import numpy as np

##### Stream function

class DynamicImageReconstructor:
    def __init__(self, min_row_length=3, max_gap=50, max_image_gap=100):
        self.min_row_length = min_row_length
        self.max_gap = max_gap
        self.max_image_gap = max_image_gap
        self.tokens = []
        self.images = []
        self.last_processed_index = -1

    def add_token(self, token):
        self.tokens.append(token)
        if isinstance(token, tuple):
            self._process_new_token(len(self.tokens) - 1)

    def _process_new_token(self, index):
        if index <= self.last_processed_index:
            return

        # Trouver le début de la nouvelle séquence à traiter
        start = max(self.last_processed_index + 1, index - self.max_gap)
        new_sequence = self.tokens[start:index+1]
        
        # Identifier les pixels valides dans la nouvelle séquence
        valid_pixels = [(i+start, pixel) for i, pixel in enumerate(new_sequence) if isinstance(pixel, tuple)]

        if not valid_pixels:
            return

        # Mettre à jour ou créer une nouvelle image
        if self.images and index - self.images[-1]['end_index'] <= self.max_gap:
            self._update_last_image(valid_pixels)
        else:
            self._create_new_image(valid_pixels)

        self.last_processed_index = index

    def _update_last_image(self, valid_pixels):
        last_image = self.images[-1]['image']
        new_rows = []
        current_row = []

        for idx, pixel in valid_pixels:
            if current_row and idx - current_row[-1][0] > 1:
                if len(current_row) >= self.min_row_length:
                    new_rows.append([p for _, p in current_row])
                current_row = []
            current_row.append((idx, pixel))

        if current_row and len(current_row) >= self.min_row_length:
            new_rows.append([p for _, p in current_row])

        if new_rows:
            max_width = max(max(len(row) for row in new_rows), last_image.shape[1])
            updated_image = np.full((last_image.shape[0] + len(new_rows), max_width, 3), 255, dtype=np.uint8)
            updated_image[:last_image.shape[0], :last_image.shape[1]] = last_image

            for i, row in enumerate(new_rows):
                updated_image[last_image.shape[0] + i, :len(row)] = row

            self.images[-1]['image'] = updated_image
            self.images[-1]['end_index'] = valid_pixels[-1][0]

    def _create_new_image(self, valid_pixels):
        rows = []
        current_row = []

        for idx, pixel in valid_pixels:
            if current_row and idx - current_row[-1][0] > 1:
                if len(current_row) >= self.min_row_length:
                    rows.append([p for _, p in current_row])
                current_row = []
            current_row.append((idx, pixel))

        if current_row and len(current_row) >= self.min_row_length:
            rows.append([p for _, p in current_row])

        if rows:
            max_width = max(len(row) for row in rows)
            new_image = np.full((len(rows), max_width, 3), 255, dtype=np.uint8)

            for i, row in enumerate(rows):
                new_image[i, :len(row)] = row

            self.images.append({
                'image': new_image,
                'start_index': valid_pixels[0][0],
                'end_index': valid_pixels[-1][0]
            })

    def get_images(self):
        return self.images

def reconstruct_imgs(tokens_arr_obj, min_row_length=3, max_gap=50, max_image_gap=100):
    # Trouver les indices des tuples (RGB)
    img_idx = np.where(np.vectorize(lambda x: isinstance(x, tuple))(tokens_arr_obj))[0]
    if len(img_idx) < min_row_length:
        return []

    # Identifier les écarts et les images
    gaps = np.diff(img_idx)
    image_splits = np.split(img_idx, np.where(gaps > max_gap)[0] + 1)

    # Créer les images
    images = [
        np.array([tokens_arr_obj[i] for i in split])
        for split in image_splits if len(split) >= min_row_length
    ]

    # Regrouper les images proches
    if not images:
        return []

    # Identifier les groupes d'images
    start_indices = np.array([split[0] for split in image_splits if len(split) >= min_row_length])
    group_breaks = np.where(np.diff(start_indices) > max_image_gap)[0] + 1
    image_groups = np.split(images, group_breaks)

    # Concaténer les images dans chaque groupe
    final_images = []
    for group in image_groups:
        if len(group) == 1:
            final_images.append({'image': group[0], 'start_index': start_indices[0], 'end_index': start_indices[0]})
        else:
            max_width = max(img.shape[1] for img in group)
            total_height = sum(img.shape[0] for img in group)
            final_image = np.full((total_height, max_width, 3), 255, dtype=np.uint8)
            current_height = 0
            for img in group:
                h, w, _ = img.shape
                final_image[current_height:current_height+h, :w] = img
                current_height += h
            final_images.append({'image': final_image, 'start_index': start_indices[0], 'end_index': start_indices[-1]})

    return final_images

def reconstruct_imgs(tokens_arr_obj, min_row_length=3, max_gap=50):
    # Trouver les indices des tuples (RGB)
    img_idx = np.where(np.vectorize(lambda x: isinstance(x, tuple))(tokens_arr_obj))[0]
    if len(img_idx) < min_row_length: return []
    # Trouver les grands écarts qui séparent les images
    big_gaps = np.where(np.diff(img_idx) > max_gap)[0] + 1
    image_splits = np.split(img_idx, big_gaps)
    images = []
    for split in image_splits:
        if len(split) < min_row_length: continue
        # Find gap between image
        row_breaks = np.concatenate(([0], np.where(np.diff(split) > 1)[0] + 1, [len(split)]))
        row_lengths = np.diff(row_breaks)
        # Row valid filter
        valid_rows = row_lengths >= min_row_length
        if not np.any(valid_rows): continue
        most_common_length = np.bincount(row_lengths[valid_rows]).argmax()
        # Create images
        num_rows = np.sum(valid_rows)
        img = np.full((num_rows, most_common_length, 3), 255, dtype=np.uint8)
        valid_indices = split[np.concatenate([np.arange(start, end) for start, end, valid 
                                              in zip(row_breaks[:-1], row_breaks[1:], valid_rows) 
                                              if valid])]
        img_flat = img.reshape(-1, 3)
        img_flat[:len(valid_indices)] = [tokens_arr_obj[i] for i in valid_indices]
        images.append({'image': img, 'start_index': split[0], 'end_index': split[-1]})
    return images