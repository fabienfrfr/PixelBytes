#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
Extract file from : https://www.pokencyclopedia.info/en/index.php?id=sprites/gen5
"""

import os
import cv2
import numpy as np
from skimage import color
import imageio

class PixelizeGIF:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.palette = np.array([
            [0x00, 0x00, 0x00], [0xfc, 0xfc, 0xfc], [0xf8, 0xf8, 0xf8], [0xbc, 0xbc, 0xbc],
            [0x7c, 0x7c, 0x7c], [0xa4, 0xe4, 0xfc], [0x3c, 0xbc, 0xfc], [0x00, 0x78, 0xf8],
            [0x00, 0x00, 0xfc], [0xb8, 0xb8, 0xf8], [0x68, 0x88, 0xfc], [0x00, 0x58, 0xf8],
            [0x00, 0x00, 0xbc], [0xd8, 0xb8, 0xf8], [0x98, 0x78, 0xf8], [0x68, 0x44, 0xfc],
            [0x44, 0x28, 0xbc], [0xf8, 0xb8, 0xf8], [0xf8, 0x78, 0xf8], [0xd8, 0x00, 0xcc],
            [0x94, 0x00, 0x84], [0xf8, 0xa4, 0xc0], [0xf8, 0x58, 0x98], [0xe4, 0x00, 0x58],
            [0xa8, 0x00, 0x20], [0xf0, 0xd0, 0xb0], [0xf8, 0x78, 0x58], [0xf8, 0x38, 0x00],
            [0xa8, 0x10, 0x00], [0xfc, 0xe0, 0xa8], [0xfc, 0xa0, 0x44], [0xe4, 0x5c, 0x10],
            [0x88, 0x14, 0x00], [0xf8, 0xd8, 0x78], [0xf8, 0xb8, 0x00], [0xac, 0x7c, 0x00],
            [0x50, 0x30, 0x00], [0xd8, 0xf8, 0x78], [0xb8, 0xf8, 0x18], [0x00, 0xb8, 0x00],
            [0x00, 0x78, 0x00], [0xb8, 0xf8, 0xb8], [0x58, 0xd8, 0x54], [0x00, 0xa8, 0x00],
            [0x00, 0x68, 0x00], [0xb8, 0xf8, 0xd8], [0x58, 0xf8, 0x98], [0x00, 0xa8, 0x44],
            [0x00, 0x58, 0x00], [0x00, 0xfc, 0xfc], [0x00, 0xe8, 0xd8], [0x00, 0x88, 0x88],
            [0x00, 0x40, 0x58], [0xf8, 0xd8, 0xf8], [0x78, 0x78, 0x78]], dtype=np.uint8)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def pixelize_frame(self, frame, max_size=50, white_threshold=250):
        if frame.shape[2] == 4:
            alpha = frame[:, :, 3] / 255.0
            frame = (alpha[:, :, np.newaxis] * frame[:, :, :3] + (1 - alpha[:, :, np.newaxis]) * 255).astype(np.uint8)
        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask = gray > white_threshold
        frame_lab = color.rgb2lab(frame)
        palette_lab = color.rgb2lab(self.palette.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        distances = np.sum((frame_lab[:, :, np.newaxis, :] - palette_lab[np.newaxis, np.newaxis, :, :])**2, axis=3)
        nearest_colors = self.palette[np.argmin(distances, axis=2)]
        result = np.where(mask[:, :, np.newaxis], 255, nearest_colors)
        return result

    def process_gif(self, file_name):
        input_path = os.path.join(self.input_folder, file_name)
        output_path = os.path.join(self.output_folder, f"{file_name}")
        with imageio.get_reader(input_path) as reader, imageio.get_writer(output_path, mode='I', duration=reader.get_meta_data().get('duration', 100)) as writer:
            i = 0
            for frame in reader:
                pixelized = self.pixelize_frame(frame)
                if i>0 : writer.append_data(pixelized)
                i+=1 #first image bug
        print(f"Saved pixelized GIF: {output_path}")

    def process_all_gifs(self):
        for file_name in os.listdir(self.input_folder):
            if file_name.endswith('.gif'):
                print(f"Processing: {file_name}")
                self.process_gif(file_name)
                print(f"Completed: {file_name}")


if __name__ == '__main__' :
    
    pixelizer = PixelizeGIF('pokemon_gifs', 'output_gifs')
    pixelizer.process_all_gifs()
    pixelizer = PixelizeGIF('pokemon_back_gifs', 'output_gifs')
    pixelizer.process_all_gifs()
    
    from datasets import load_dataset
    from huggingface_hub import login

    def push_dataset(dataset, repo_name="ffurfaro/PixelBytes-PokemonSprites"):
        token = input("Input Hugging Face Token: ")
        # Connect and push to Hub
        login(token)
        dataset.push_to_hub(repo_name)
    
    dataset = load_dataset("imagefolder", data_dir="output_gifs")
    push_dataset(dataset)