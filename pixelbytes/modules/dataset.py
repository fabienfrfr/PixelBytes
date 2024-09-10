#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

from .config import DEFAULT_PALETTE
from .tokenizer import PixelBytesTokenizer

import getpass
from huggingface_hub import login

from torch.utils.data import Dataset
import torch

import numpy as np, cv2
from numpy.lib.stride_tricks import as_strided
from skimage import color
from tqdm import tqdm

##### dataset
class PxByDataset(Dataset):
    def __init__(self, pxby_columns, seq_length=128, stride=32):
        self.data = [torch.tensor(sequence, dtype=torch.long) for sequence in pxby_columns]
        self.seq_length = seq_length
        self.stride = stride
        self.sub_sequences = self._create_sub_sequences()

    def _create_sub_sequences(self):
        sub_sequences = []
        for sequence in self.data:
            L, H, W = sequence.shape
            # Define index tensor for all subseq
            starts = torch.arange(0, L, self.stride)
            indices = (starts[:, None] + torch.arange(self.seq_length + 1)) % L
            # Extract and construct all input-target
            sub_seqs = sequence[indices]
            inputs = sub_seqs[:, :-1]
            targets = sub_seqs[:, -1, H//2, W//2]
            sub_sequences.extend(zip(inputs, targets))
        return sub_sequences
    
    def __len__(self):
        return len(self.sub_sequences)

    def __getitem__(self, idx):
        return self.sub_sequences[idx]

def push_dataset(dataset, repo_name="ffurfaro/PixelBytes-Pokemon"):
    token = getpass.getpass("Input Hugging Face Token: ")
    # Connect and push to Hub
    login(token)
    dataset.push_to_hub(repo_name)

## Construct pixelbytes columns
def input_seq_construct(arr, dim=3, none_val=0, pix_sep=1, modal_sep=2):
    if dim % 2 == 0 : dim +=1
    pw = (dim-1)//2
    # add separator in the end
    h,w = arr.shape
    separator = pix_sep*np.ones((h, 1), dtype=np.uint8)
    separator[-1,-1] = modal_sep
    arr = np.concatenate((arr, separator), axis=1)
    # update
    shape = (h, w+1, dim, dim)
    # Padding around array and size
    padded_arr = np.pad(arr, pad_width=pw, mode='constant', constant_values=0)
    strides = padded_arr.strides * 2  # double step
    # Data matrix construct
    matrix_dxd = as_strided(padded_arr, shape=shape, strides=strides)
    # include time asymetry (need correction for dim!=3)
    result = np.copy(matrix_dxd).astype(np.uint8)
    result[:, :, pw:, -pw:] = none_val
    result[:, :, -pw, :] = none_val
    # flatten
    return result.reshape(-1, dim, dim)

def add_pixelbyte_columns(image_caption_dataset, is_pixel_style=False, tokenizer=PixelBytesTokenizer(), palette=DEFAULT_PALETTE):
    vocab = tokenizer.get_vocab()
    # vocabulary
    n = vocab[b'\x00']
    p = vocab[b'\t']
    m = vocab[b'\n']
    # image map init
    vectorized_map = np.vectorize(lambda x,y,z: vocab.get((int(x), int(y), int(z)), None))
    # add pixelbytes columns
    pixelbytes = []
    for row in tqdm(image_caption_dataset, desc="Construct pixelbytes columns"):
        # get info
        Image = np.array(row['image'])
        Caption = row['caption'].encode('utf-8')
        ### Image part
        if is_pixel_style : Image = image_paletization(Image, palette) # already pixelized style
        else : Image = image_pixelization(Image, palette) # Pixel reduce & palettize 
        # Separate RGB channel & Quantize
        Image = vectorized_map(Image[..., 0], Image[..., 1], Image[..., 2])
        # create sequence image
        Image = input_seq_construct(Image, dim=3, none_val=n, pix_sep=p, modal_sep=m)
        ### Shape part
        Shape = str(Image.shape[:-1]).encode('utf-8')
        Shape = np.array([vocab[bytes([x])] for x in Shape])[None]
        Shape = input_seq_construct(Shape, dim=3, none_val=n, pix_sep=p, modal_sep=m)
        ### Caption part
        Caption = np.array([vocab[bytes([x])] for x in Caption])[None]
        Caption = input_seq_construct(Caption, dim=3, none_val=n, pix_sep=p, modal_sep=m)
        # Combine
        pixelbyte = np.concatenate((Shape, Image, Caption), axis=0)
        pixelbytes.append(pixelbyte.tolist())
    # return new column
    if "pixelbyte" in image_caption_dataset.column_names : 
        image_caption_dataset = image_caption_dataset.remove_columns("pixelbyte")
    return image_caption_dataset.add_column("pixelbyte", pixelbytes)

##### Image function
def image_autocrop(img) :
    # Binarize image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Find Rectangle
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Crop image
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)
    return img[y_min:y_max, x_min:x_max]

def alpha_to_blank(img) :
    rgb_channels = img[:, :, :3]
    alpha_channel = img[:, :, 3] / 255.0
    # Merge image with white background
    white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
    return cv2.convertScaleAbs(rgb_channels * alpha_channel[..., None] + white_background * (1 - alpha_channel[..., None]))

def resize_image(img, h, w, m=25, a=1) :
    scale_factor = m / max(h, w)
    # resizing
    new_height = int(min(h,w) * scale_factor)
    return cv2.resize(img, (a*new_height, a*m) if h>w else (a*m, 2*new_height), interpolation=cv2.INTER_NEAREST)

def kmean_quantization(img, num_colors) :
    # Flatten image and convertion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    # K-Means Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Change pixel following cluster
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    return cv2.cvtColor(quantized.reshape(img.shape), cv2.COLOR_Lab2BGR)

def image_edging_enhancer(img) :
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    # kernel & dilatation
    kernel_size = int(min(img.shape[0], img.shape[1]) * 0.005)  # 0,5% of image size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_edges = cv2.dilate(edges, kernel)
    # border dark coloring
    img_ = img.copy()
    img_[dilated_edges != 0] = [0, 0, 0]
    return img_

def image_paletization(img, palette) :
    palette_rgb = palette.copy()
    palette = color.rgb2lab(palette_rgb.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    def closest_color(pixel, palette):
        pixel_lab = color.rgb2lab(pixel.reshape(1, 1, 3) / 255.0).reshape(3)
        # distance
        distances = np.linalg.norm(palette - pixel_lab, axis=1)
        # Retourner la couleur la plus proche en RGB
        return palette_rgb[np.argmin(distances)]
    return np.apply_along_axis(closest_color, 2 , img, palette)
    
def image_pixelization(img, palette, max_size=25, edging=True):
    # Parameter
    h, w = img.shape[:2]
    num_colors = len(palette)
    ## Cropping part
    img = image_autocrop(img)
    ## Pretreatment
    # Remove alpha (if)
    if img.shape[2] == 4:
        img = alpha_to_blank(img)
    # edging (experimental for big img)
    img = image_edging_enhancer(img) if edging and min(h,w)>200 else img
    # First resizing (a=2)
    img = resize_image(img, h, w, m=max_size, a=2)
    ## Quantize image
    img = kmean_quantization(img, num_colors)
    # resizing
    img = resize_image(img, h, w, m=max_size)
    ## Palette association
    return image_paletization(img, palette)